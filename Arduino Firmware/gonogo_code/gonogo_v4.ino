#include <Arduino.h>
#include <RTClib.h>
#include "MouseHouse.h"

// TASK_INFO,unix_us,rp2040_us,nan,phase,hold_us,go_us,go_prob,context,reason
// NOGO_STAGE_INFO,unix_us,rp2040_us,nan,nogoStage,nogo_us,eval_goal,go_prob,context,reason

MouseHouse mh;
RTC_DS3231 rtc;

// Phase timing + reward
const uint32_t GO_TONE_FREQ = 16000;
const uint32_t NOGO_TONE_FREQ = 8000;
const uint32_t GONOGO_TONE_DURATION_MS = 250;
const uint32_t GO_TONE_DURATION_MS = 150;
const uint32_t GO_TONE_CLICK_ON_MS = 5;
const uint32_t GO_TONE_CLICK_OFF_MS = 5;
uint32_t GO_TIME_MS = 300;
uint32_t NOGO_TIME_MS = 300;
uint32_t TRIAL_START_HOLD_MS = 50;
const uint32_t TIMEOUT_MS = 600;
const int REWARD_FEED_STEPS = 1024;
const uint32_t CLICK_FREQ = 2000;
const uint32_t CLICK_ON_MS = 50;
const uint32_t CLICK_OFF_MS = 50;
const int CLICK_TOTAL = 3;

// Task + sides
const int TRIAL_GO = 1;
const int TRIAL_NOGO = 0;
const int SIDE_LEFT = 0;
const int SIDE_RIGHT = 1;
const int PHASE_1_FR1 = 1;
const int PHASE_2_HOLD_CUE = 2;
const int PHASE_3_NOGO = 3;
const int PHASE_4_GO_MIX = 4;

// LED colors / pixels
const uint8_t TRIAL_STRIP_R = 0;
const uint8_t TRIAL_STRIP_G = 0;
const uint8_t TRIAL_STRIP_B = 0;
const uint8_t TRIAL_STRIP_W = 5;
const uint8_t LEFT_POKE_R = 0;
const uint8_t LEFT_POKE_G = 0;
const uint8_t LEFT_POKE_B = 50;
const uint8_t LEFT_POKE_W = 0;
const uint8_t RIGHT_POKE_R = 0;
const uint8_t RIGHT_POKE_G = 0;
const uint8_t RIGHT_POKE_B = 50;
const uint8_t RIGHT_POKE_W = 0;

// Phase config
const int START_PHASE = PHASE_3_NOGO;
const unsigned long PHASE_1_PELLET_THRESH = 50;
const uint32_t PHASE_1_INIT_HOLD_MS = 50;
const unsigned long PHASE_2_REWARDS_PER_HOLD_STEP = 20;
const uint32_t PHASE_2_HOLD_INCREMENT_MS = 25;
const uint32_t PHASE_2_HOLD_TARGET_MS = 200;
const int INIT_WINDOW = 40;
const float INIT_RATIO_THRESHOLD = 0.5f;
const uint32_t PHASE_2_HOLD_REGRESS_STEP_MS = 25;
const uint32_t PHASE_3_NOGO_STAGE_TIMES_MS[] = { 200, 250, 300, 350, 400, 450, 500 };
const uint8_t PHASE_3_GO_PROB_PERCENT = 50;
const uint32_t PHASE_3_GO_TIME_MS = 200;
const int PHASE_3_NOGO_STAGE_COUNT = 7;
const unsigned long PHASE_3_REWARDS_PER_STAGE = 20;
const float PHASE_3_ACCURACY_THRESHOLD = 0.6f;
const uint32_t PHASE_3_NOGO_REGRESS_STEP_MS = 25;
const uint8_t PHASE_4_GO_PROB_PERCENT = 50;

// Timestamp helpers
uint64_t base_us = 0;
uint64_t base_unix_us = 0;

// Runtime state
bool sessionWasRunning = false;
bool phaseInitialized = false;
bool goTimeoutTriggered = false;
bool waitingForRelease = false;
bool rewardFeedStarting = false;
uint64_t trialCueStart_us = 0;
uint64_t leftPokeStartUnix = 0;
uint64_t rightPokeStartUnix = 0;
bool prevLeftPokeActive = false;
bool prevRightPokeActive = false;
bool prevFeedActive = false;
bool trialAvailabilityLit = false;
int trialAvailabilitySide = -1;
bool leftPokeEligible = false;
bool rightPokeEligible = false;

int trainingPhase = START_PHASE;
int currentActiveSide = SIDE_LEFT;
int currentTrialType = TRIAL_GO;
int waitingForReleaseSide = -1;
int nogoStage = 0;
unsigned long sideSetCount = 0;
unsigned long trialCount = 0;
unsigned long rewardFeedCount = 0;
unsigned long phaseFeedBase = 0;
unsigned long activePokeCount = 0;
unsigned long trialStartCount = 0;
unsigned long noGoTrialCount = 0;
unsigned long noGoCorrectCount = 0;

unsigned long activePokeSnapshot[INIT_WINDOW];
unsigned long trialStartSnapshot[INIT_WINDOW];
unsigned long noGoTrialSnapshot[INIT_WINDOW];
unsigned long noGoCorrectSnapshot[INIT_WINDOW];
int snapshotIndex = 0;
int snapshotCount = 0;
int noGoSnapshotIndex = 0;
int noGoSnapshotCount = 0;
uint8_t goProbPercent = 0;

bool trialActive = false;

#define NAN_STR "nan"
#define NAN_INT 69420

extern "C" uint64_t time_us_64();
const char* getPokeContext();

uint64_t get_timestamp_us() {
  uint64_t elapsed_us = time_us_64() - base_us;
  return base_unix_us + elapsed_us;
}

void logEvent(const char* eventType,
              uint64_t unixTime,
              uint64_t rp2040Time,
              const char* side,
              unsigned long count,
              uint64_t duration,
              uint64_t latency,
              long value,
              const char* context,
              const char* reason) {
  Serial.print(eventType);
  Serial.print(",");
  Serial.print(unixTime);
  Serial.print(",");
  Serial.print(rp2040Time);
  Serial.print(",");
  Serial.print(side);
  Serial.print(",");
  Serial.print(count);
  Serial.print(",");
  Serial.print(duration);
  Serial.print(",");
  Serial.print(latency);
  Serial.print(",");
  Serial.print(value);
  Serial.print(",");
  Serial.print(context);
  Serial.print(",");
  Serial.println(reason);
}

void syncLibraryTaskContext() {
  mh.setTaskContext(getPokeContext());
}

const char* getPokeContext() {
  if (mh.isTimeoutActive()) {
    return "Timeout";
  } else if (mh.isPelletAvailable() || mh.pelletSensorBlocked()) {
    return "Pellet_Available";
  } else if (rewardFeedStarting || mh.isFeedActive()) {
    return "Feeding";
  } else if (waitingForRelease) {
    return "Waiting_For_Release";
  }
  return "Eligible";
}

const char* getTrialLabel(int trialType) {
  return (trialType == TRIAL_GO) ? "GO" : "NOGO";
}

const char* getPhaseTrialReason() {
  if (trainingPhase == PHASE_1_FR1) return "PHASE1";
  if (trainingPhase == PHASE_2_HOLD_CUE) return "PHASE2";
  return getTrialLabel(currentTrialType);
}

int pickTrialType() {
  if (goProbPercent >= 100) return TRIAL_GO;
  if (goProbPercent == 0) return TRIAL_NOGO;
  return (random(100) < goProbPercent) ? TRIAL_GO : TRIAL_NOGO;
}

bool isInitiationEligible() {
  if (mh.isTimeoutActive() || rewardFeedStarting || mh.isFeedActive()) return false;
  if (mh.isPelletAvailable()) return false;
  if (mh.pelletSensorBlocked()) return false;
  if (waitingForRelease) return false;
  return true;
}

void resetInitiationWindow() {
  activePokeCount = 0;
  trialStartCount = 0;
  snapshotIndex = 0;
  snapshotCount = 0;
}

void recordPokeSnapshot() {
  activePokeSnapshot[snapshotIndex] = activePokeCount;
  trialStartSnapshot[snapshotIndex] = trialStartCount;
  snapshotIndex = (snapshotIndex + 1) % INIT_WINDOW;
  if (snapshotCount < INIT_WINDOW) snapshotCount++;
}

bool computeInitiationRatio(float* ratioOut) {
  if (snapshotCount < INIT_WINDOW) return false;
  int oldest = snapshotIndex;
  unsigned long deltaPokes = activePokeCount - activePokeSnapshot[oldest];
  unsigned long deltaTrials = trialStartCount - trialStartSnapshot[oldest];
  if (deltaPokes == 0) return false;
  *ratioOut = (float)deltaTrials / (float)deltaPokes;
  return true;
}

void resetNoGoAccuracyWindow() {
  noGoTrialCount = 0;
  noGoCorrectCount = 0;
  noGoSnapshotIndex = 0;
  noGoSnapshotCount = 0;
}

void recordNoGoSnapshot() {
  noGoTrialSnapshot[noGoSnapshotIndex] = noGoTrialCount;
  noGoCorrectSnapshot[noGoSnapshotIndex] = noGoCorrectCount;
  noGoSnapshotIndex = (noGoSnapshotIndex + 1) % INIT_WINDOW;
  if (noGoSnapshotCount < INIT_WINDOW) noGoSnapshotCount++;
}

bool computeNoGoAccuracy(float* accuracyOut) {
  if (noGoSnapshotCount < INIT_WINDOW) return false;
  int oldest = noGoSnapshotIndex;
  unsigned long deltaTrials = noGoTrialCount - noGoTrialSnapshot[oldest];
  unsigned long deltaCorrect = noGoCorrectCount - noGoCorrectSnapshot[oldest];
  if (deltaTrials == 0) return false;
  *accuracyOut = (float)deltaCorrect / (float)deltaTrials;
  return true;
}

void logTaskInfo(const char* reason) {
  logEvent("TASK_INFO",
           get_timestamp_us(), time_us_64(),
           NAN_STR,
           trainingPhase,
           TRIAL_START_HOLD_MS * 1000ULL,
           GO_TIME_MS * 1000ULL,
           goProbPercent,
           getPokeContext(),
           reason);
}

void logNoGoStageInfo(const char* reason) {
  logEvent("NOGO_STAGE_INFO",
           get_timestamp_us(), time_us_64(),
           NAN_STR,
           nogoStage,
           NOGO_TIME_MS * 1000ULL,
           PHASE_3_REWARDS_PER_STAGE,
           goProbPercent,
           getPokeContext(),
           reason);
}

void logTrialAvailabilityDebug(const char* reason) {
  bool pelletAvailable = mh.isPelletAvailable();
  bool sensorBlocked = mh.pelletSensorBlocked();
  long blockCode = (pelletAvailable ? 1 : 0) |
                   (sensorBlocked ? 2 : 0) |
                   (waitingForRelease ? 4 : 0);

  logEvent("TRIAL_AVAIL_DEBUG",
           get_timestamp_us(), time_us_64(),
           (currentActiveSide == SIDE_LEFT) ? "L" : "R",
           pelletAvailable ? 1 : 0,
           sensorBlocked ? 1 : 0,
           waitingForRelease ? 1 : 0,
           blockCode,
           getPokeContext(),
           reason);
}

void updatePhaseGoProbability() {
  if (trainingPhase == PHASE_3_NOGO) {
    goProbPercent = PHASE_3_GO_PROB_PERCENT;
  } else if (trainingPhase == PHASE_4_GO_MIX) {
    goProbPercent = PHASE_4_GO_PROB_PERCENT;
  } else {
    goProbPercent = 0;
  }
}

int sanitizeStartPhase(int requestedPhase) {
  if (requestedPhase < PHASE_1_FR1 || requestedPhase > PHASE_4_GO_MIX) {
    return PHASE_1_FR1;
  }
  return requestedPhase;
}

void enterPhase(int newPhase, const char* reason) {
  trainingPhase = newPhase;
  phaseFeedBase = rewardFeedCount;
  resetInitiationWindow();
  resetNoGoAccuracyWindow();
  waitingForRelease = false;
  waitingForReleaseSide = -1;
  goTimeoutTriggered = false;

  if (trainingPhase == PHASE_1_FR1) {
    GO_TIME_MS = PHASE_3_GO_TIME_MS;
    NOGO_TIME_MS = PHASE_3_NOGO_STAGE_TIMES_MS[0];
    TRIAL_START_HOLD_MS = PHASE_1_INIT_HOLD_MS;
    nogoStage = 0;
  } else if (trainingPhase == PHASE_2_HOLD_CUE) {
    GO_TIME_MS = PHASE_3_GO_TIME_MS;
    NOGO_TIME_MS = PHASE_3_NOGO_STAGE_TIMES_MS[0];
    TRIAL_START_HOLD_MS = PHASE_1_INIT_HOLD_MS;
    nogoStage = 0;
  } else if (trainingPhase == PHASE_3_NOGO) {
    GO_TIME_MS = PHASE_3_GO_TIME_MS;
    TRIAL_START_HOLD_MS = PHASE_2_HOLD_TARGET_MS;
    nogoStage = 0;
    NOGO_TIME_MS = PHASE_3_NOGO_STAGE_TIMES_MS[nogoStage];
  } else if (trainingPhase == PHASE_4_GO_MIX) {
    TRIAL_START_HOLD_MS = PHASE_2_HOLD_TARGET_MS;
    NOGO_TIME_MS = PHASE_3_NOGO_STAGE_TIMES_MS[PHASE_3_NOGO_STAGE_COUNT - 1];
    nogoStage = PHASE_3_NOGO_STAGE_COUNT - 1;
    GO_TIME_MS = PHASE_3_GO_TIME_MS;
  }

  updatePhaseGoProbability();
  currentTrialType = pickTrialType();
  logTaskInfo(reason);
  if (trainingPhase >= PHASE_3_NOGO) {
    logNoGoStageInfo(reason);
  }
}

void initializePhaseIfNeeded() {
  if (phaseInitialized) return;
  phaseInitialized = true;
  enterPhase(sanitizeStartPhase(START_PHASE), "PhaseInit");
}

void maybeAdvancePhase2HoldOnRewardStart() {
  if (trainingPhase != PHASE_2_HOLD_CUE) return;
  if (TRIAL_START_HOLD_MS >= PHASE_2_HOLD_TARGET_MS) return;
  if ((rewardFeedCount - phaseFeedBase) < PHASE_2_REWARDS_PER_HOLD_STEP) return;

  float ratio = 0.0f;
  if (!computeInitiationRatio(&ratio)) return;
  if (ratio < INIT_RATIO_THRESHOLD) return;

  TRIAL_START_HOLD_MS = (TRIAL_START_HOLD_MS + PHASE_2_HOLD_INCREMENT_MS > PHASE_2_HOLD_TARGET_MS)
                          ? PHASE_2_HOLD_TARGET_MS
                          : (TRIAL_START_HOLD_MS + PHASE_2_HOLD_INCREMENT_MS);
  phaseFeedBase += PHASE_2_REWARDS_PER_HOLD_STEP;
  resetInitiationWindow();
  logTaskInfo("Phase2Advance");
  if (TRIAL_START_HOLD_MS >= PHASE_2_HOLD_TARGET_MS) {
    enterPhase(PHASE_3_NOGO, "Phase3Start");
  }
}

void maybeRegressPhase2HoldOnEligiblePokeEnd() {
  if (trainingPhase != PHASE_2_HOLD_CUE) return;
  if (TRIAL_START_HOLD_MS <= 50) return;

  float ratio = 0.0f;
  if (!computeInitiationRatio(&ratio)) return;
  if (ratio >= INIT_RATIO_THRESHOLD) return;

  uint32_t newHold = (TRIAL_START_HOLD_MS > PHASE_2_HOLD_REGRESS_STEP_MS)
                       ? (TRIAL_START_HOLD_MS - PHASE_2_HOLD_REGRESS_STEP_MS)
                       : TRIAL_START_HOLD_MS;
  if (newHold < 50) newHold = 50;
  if (newHold == TRIAL_START_HOLD_MS) return;

  TRIAL_START_HOLD_MS = newHold;
  phaseFeedBase = rewardFeedCount;
  resetInitiationWindow();
  logTaskInfo("Phase2Regress");
}

void maybeRegressPhase3NoGoOnEligiblePokeEnd() {
  if (trainingPhase != PHASE_3_NOGO) return;
  if (NOGO_TIME_MS <= PHASE_3_NOGO_STAGE_TIMES_MS[0]) return;

  float ratio = 0.0f;
  float accuracy = 0.0f;
  if (!computeInitiationRatio(&ratio)) return;
  if (!computeNoGoAccuracy(&accuracy)) return;
  if (ratio >= INIT_RATIO_THRESHOLD && accuracy >= PHASE_3_ACCURACY_THRESHOLD) return;

  uint32_t newNoGo = (NOGO_TIME_MS > PHASE_3_NOGO_REGRESS_STEP_MS)
                       ? (NOGO_TIME_MS - PHASE_3_NOGO_REGRESS_STEP_MS)
                       : NOGO_TIME_MS;
  if (newNoGo < PHASE_3_NOGO_STAGE_TIMES_MS[0]) {
    newNoGo = PHASE_3_NOGO_STAGE_TIMES_MS[0];
  }
  if (newNoGo == NOGO_TIME_MS) return;

  NOGO_TIME_MS = newNoGo;
  while (nogoStage > 0 && PHASE_3_NOGO_STAGE_TIMES_MS[nogoStage] > NOGO_TIME_MS) {
    nogoStage--;
  }
  updatePhaseGoProbability();
  phaseFeedBase = rewardFeedCount;
  resetInitiationWindow();
  resetNoGoAccuracyWindow();
  logTaskInfo("Phase3Regress");
  logNoGoStageInfo("Phase3Regress");
}

void maybeAdjustPhase3NoGoOnRewardStart() {
  if (trainingPhase != PHASE_3_NOGO) return;
  if ((rewardFeedCount - phaseFeedBase) < PHASE_3_REWARDS_PER_STAGE) return;

  float ratio = 0.0f;
  float accuracy = 0.0f;
  if (!computeInitiationRatio(&ratio)) return;
  if (!computeNoGoAccuracy(&accuracy)) return;

  if (ratio >= INIT_RATIO_THRESHOLD && accuracy >= PHASE_3_ACCURACY_THRESHOLD) {
    if (nogoStage < (PHASE_3_NOGO_STAGE_COUNT - 1)) {
      nogoStage++;
      NOGO_TIME_MS = PHASE_3_NOGO_STAGE_TIMES_MS[nogoStage];
      updatePhaseGoProbability();
      phaseFeedBase = rewardFeedCount;
      resetInitiationWindow();
      resetNoGoAccuracyWindow();
      logTaskInfo("Phase3Advance");
      logNoGoStageInfo("Phase3Advance");
      return;
    }

    if (nogoStage >= (PHASE_3_NOGO_STAGE_COUNT - 1)) {
      enterPhase(PHASE_4_GO_MIX, "Phase4Start");
    }
    return;
  }

  if (NOGO_TIME_MS > PHASE_3_NOGO_STAGE_TIMES_MS[0]) {
    uint32_t newNoGo = (NOGO_TIME_MS > PHASE_3_NOGO_REGRESS_STEP_MS)
                         ? (NOGO_TIME_MS - PHASE_3_NOGO_REGRESS_STEP_MS)
                         : NOGO_TIME_MS;
    if (newNoGo < PHASE_3_NOGO_STAGE_TIMES_MS[0]) {
      newNoGo = PHASE_3_NOGO_STAGE_TIMES_MS[0];
    }
    NOGO_TIME_MS = newNoGo;
    while (nogoStage > 0 && PHASE_3_NOGO_STAGE_TIMES_MS[nogoStage] > NOGO_TIME_MS) {
      nogoStage--;
    }
    updatePhaseGoProbability();
    logTaskInfo("Phase3Regress");
    logNoGoStageInfo("Phase3Regress");
  }

  phaseFeedBase = rewardFeedCount;
  resetInitiationWindow();
  resetNoGoAccuracyWindow();
}

void registerRewardFeedStart() {
  rewardFeedStarting = true;
  rewardFeedCount++;

  if (trainingPhase == PHASE_1_FR1 && (rewardFeedCount - phaseFeedBase) >= PHASE_1_PELLET_THRESH) {
    enterPhase(PHASE_2_HOLD_CUE, "Phase2Start");
  } else if (trainingPhase == PHASE_2_HOLD_CUE) {
    maybeAdvancePhase2HoldOnRewardStart();
  } else if (trainingPhase == PHASE_3_NOGO) {
    maybeAdjustPhase3NoGoOnRewardStart();
  }

  syncLibraryTaskContext();
  mh.feed(REWARD_FEED_STEPS);
  rewardFeedStarting = false;
}

void set_trial_available_off() {
  syncLibraryTaskContext();
  mh.clearMainStrip();
  mh.leftPokeLightOff();
  mh.rightPokeLightOff();
  trialAvailabilityLit = false;
  trialAvailabilitySide = -1;
}

void set_trial_available_on() {
  if (mh.isPelletAvailable() || mh.pelletSensorBlocked() || waitingForRelease) {
    logTrialAvailabilityDebug("Blocked");
    set_trial_available_off();
    return;
  }

  if (trialAvailabilityLit && trialAvailabilitySide == currentActiveSide) {
    return;
  }

  syncLibraryTaskContext();
  mh.setMainStrip(TRIAL_STRIP_R, TRIAL_STRIP_G, TRIAL_STRIP_B, TRIAL_STRIP_W);
  if (currentActiveSide == SIDE_LEFT) {
    mh.leftPokeLightOn(LEFT_POKE_R, LEFT_POKE_G, LEFT_POKE_B, LEFT_POKE_W);
    mh.rightPokeLightOff();
  } else {
    mh.leftPokeLightOff();
    mh.rightPokeLightOn(RIGHT_POKE_R, RIGHT_POKE_G, RIGHT_POKE_B, RIGHT_POKE_W);
  }
  trialAvailabilityLit = true;
  trialAvailabilitySide = currentActiveSide;
  logTrialAvailabilityDebug("On");
}

void startTaskTimeout(const char* reason) {
  trialAvailabilityLit = false;
  trialAvailabilitySide = -1;
  syncLibraryTaskContext();
  mh.startTimeout(TIMEOUT_MS, reason,
                  MouseHouse::TIMEOUT_CLICK_PATTERN,
                  CLICK_FREQ, CLICK_ON_MS, CLICK_OFF_MS, CLICK_TOTAL);
}

void finalizeResolvedTrial(int side) {
  trialActive = false;
  waitingForRelease = true;
  waitingForReleaseSide = side;
  goTimeoutTriggered = false;
  trialCueStart_us = 0;
  currentTrialType = pickTrialType();
}

void logTrialResultNow(int side, uint64_t eventUnix, uint64_t duration_us, bool correct) {
  logEvent("TRIAL_RESULT",
           eventUnix, time_us_64(),
           (side == SIDE_LEFT) ? "L" : "R",
           trialCount,
           duration_us,
           NAN_INT,
           correct ? 1 : 0,
           getPokeContext(),
           getPhaseTrialReason());
}

void rewardTrialNow(int side, uint64_t eventUnix, uint64_t duration_us) {
  if (trainingPhase == PHASE_3_NOGO ||
      (trainingPhase == PHASE_4_GO_MIX && currentTrialType == TRIAL_NOGO)) {
    noGoTrialCount++;
    noGoCorrectCount++;
    recordNoGoSnapshot();
  }

  logTrialResultNow(side, eventUnix, duration_us, true);
  registerRewardFeedStart();
  finalizeResolvedTrial(side);
}

void rewardGoTrialOnPokeEnd(int side, uint64_t eventUnix, uint64_t duration_us) {
  logTrialResultNow(side, eventUnix, duration_us, true);
  registerRewardFeedStart();
  trialActive = false;
  goTimeoutTriggered = false;
  trialCueStart_us = 0;
  currentTrialType = pickTrialType();
}

void startTrialIfEligible(int side, bool pokeNow, uint64_t pokeStartUnix, uint64_t now_unix, uint64_t now_us) {
  if (!pokeNow || trialActive || mh.isTimeoutActive()) return;
  if (!isInitiationEligible()) return;
  if (currentActiveSide != side) return;
  if ((now_unix - pokeStartUnix) < (TRIAL_START_HOLD_MS * 1000ULL)) return;

  trialActive = true;
  trialCount++;
  trialStartCount++;
  trialCueStart_us = now_us;
  set_trial_available_off();
  goTimeoutTriggered = false;

  logEvent("TRIAL_START",
           now_unix, now_us,
           (side == SIDE_LEFT) ? "L" : "R",
           trialCount,
           NAN_INT,
           NAN_INT,
           currentTrialType,
           getPokeContext(),
           getPhaseTrialReason());

  if (trainingPhase == PHASE_1_FR1) {
    syncLibraryTaskContext();
    mh.playTone(NOGO_TONE_FREQ, GONOGO_TONE_DURATION_MS);
    rewardTrialNow(side, now_unix, now_unix - pokeStartUnix);
    return;
  }

  if (trainingPhase == PHASE_2_HOLD_CUE) {
    syncLibraryTaskContext();
    mh.playTone(NOGO_TONE_FREQ, GONOGO_TONE_DURATION_MS);
    rewardTrialNow(side, now_unix, now_unix - pokeStartUnix);
    return;
  }

  if (currentTrialType == TRIAL_GO) {
    syncLibraryTaskContext();
    mh.playPulsedTone(GO_TONE_FREQ,
                      GO_TONE_DURATION_MS,
                      GO_TONE_CLICK_ON_MS,
                      GO_TONE_CLICK_OFF_MS);
  } else {
    syncLibraryTaskContext();
    mh.playTone(NOGO_TONE_FREQ, GONOGO_TONE_DURATION_MS);
  }
}

void syncPokeStatesToCurrent() {
  prevLeftPokeActive = mh.leftPokeActive();
  prevRightPokeActive = mh.rightPokeActive();
  prevFeedActive = mh.isFeedActive();
  uint64_t nowUnix = get_timestamp_us();
  leftPokeStartUnix = prevLeftPokeActive ? nowUnix : 0;
  rightPokeStartUnix = prevRightPokeActive ? nowUnix : 0;
  leftPokeEligible = false;
  rightPokeEligible = false;
}

void handlePokeStartEdges() {
  bool leftNow = mh.leftPokeActive();
  bool rightNow = mh.rightPokeActive();
  uint64_t nowUnix = get_timestamp_us();

  if (leftNow && !prevLeftPokeActive) {
    leftPokeStartUnix = nowUnix;
    leftPokeEligible = (currentActiveSide == SIDE_LEFT && isInitiationEligible());
    if (leftPokeEligible) {
      activePokeCount++;
    }
  }

  if (rightNow && !prevRightPokeActive) {
    rightPokeStartUnix = nowUnix;
    rightPokeEligible = (currentActiveSide == SIDE_RIGHT && isInitiationEligible());
    if (rightPokeEligible) {
      activePokeCount++;
    }
  }
}

void updateTrialWhilePokeHeld() {
  bool leftNow = mh.leftPokeActive();
  bool rightNow = mh.rightPokeActive();
  uint64_t nowUnix = get_timestamp_us();
  uint64_t nowUs = time_us_64();

  startTrialIfEligible(SIDE_LEFT, leftNow, leftPokeStartUnix, nowUnix, nowUs);
  startTrialIfEligible(SIDE_RIGHT, rightNow, rightPokeStartUnix, nowUnix, nowUs);

  if (!trialActive) return;

  bool activeNow = (currentActiveSide == SIDE_LEFT) ? leftNow : rightNow;
  uint64_t activePokeStartUnix = (currentActiveSide == SIDE_LEFT) ? leftPokeStartUnix : rightPokeStartUnix;

  if ((trainingPhase == PHASE_3_NOGO || trainingPhase == PHASE_4_GO_MIX) &&
      currentTrialType == TRIAL_NOGO && activeNow) {
    uint64_t cueHoldDuration = time_us_64() - trialCueStart_us;
    if (cueHoldDuration >= (NOGO_TIME_MS * 1000ULL)) {
      rewardTrialNow(currentActiveSide, nowUnix, nowUnix - activePokeStartUnix);
      return;
    }
  }

  if ((trainingPhase == PHASE_3_NOGO || trainingPhase == PHASE_4_GO_MIX) &&
      currentTrialType == TRIAL_GO && activeNow && !goTimeoutTriggered) {
    uint64_t cueElapsed = time_us_64() - trialCueStart_us;
    if (cueElapsed >= (GO_TIME_MS * 1000ULL)) {
      goTimeoutTriggered = true;
      startTaskTimeout("GoHoldExceeded");
    }
  }
}

void resolveTrialOnPokeEnd(int side, uint64_t pokeEndUnix, uint64_t pokeStartUnix) {
  if (!trialActive || currentActiveSide != side || (mh.isTimeoutActive() && !goTimeoutTriggered)) {
    return;
  }

  uint64_t duration_us = pokeEndUnix - pokeStartUnix;
  bool correct = false;

  if ((trainingPhase == PHASE_3_NOGO || trainingPhase == PHASE_4_GO_MIX) && currentTrialType == TRIAL_GO) {
    uint64_t cueElapsed = time_us_64() - trialCueStart_us;
    correct = (cueElapsed < (GO_TIME_MS * 1000ULL));
  } else if (currentTrialType == TRIAL_NOGO) {
    uint64_t cueElapsed = time_us_64() - trialCueStart_us;
    correct = (cueElapsed > (NOGO_TIME_MS * 1000ULL));
  }

  if (currentTrialType == TRIAL_NOGO) {
    noGoTrialCount++;
    recordNoGoSnapshot();
    maybeRegressPhase3NoGoOnEligiblePokeEnd();
  }

  if (correct && (trainingPhase == PHASE_3_NOGO || trainingPhase == PHASE_4_GO_MIX) && currentTrialType == TRIAL_GO) {
    rewardGoTrialOnPokeEnd(side, pokeEndUnix, duration_us);
    return;
  }

  if (!correct && !goTimeoutTriggered) {
    startTaskTimeout(getTrialLabel(currentTrialType));
  }

  logTrialResultNow(side, pokeEndUnix, duration_us, correct);

  trialActive = false;
  goTimeoutTriggered = false;
  trialCueStart_us = 0;
  currentTrialType = pickTrialType();
}

void handleOnePokeEnd(int side, bool endedNow, uint64_t pokeStartUnix, bool* eligibleFlag) {
  if (!endedNow) return;

  uint64_t pokeEndUnix = get_timestamp_us();
  uint64_t duration = pokeEndUnix - pokeStartUnix;

  if (waitingForRelease && waitingForReleaseSide == side) {
    waitingForRelease = false;
    waitingForReleaseSide = -1;
    if (mh.isRunning() && isInitiationEligible()) {
      set_trial_available_on();
    }
  }

  if (*eligibleFlag) {
    recordPokeSnapshot();
    maybeRegressPhase2HoldOnEligiblePokeEnd();
    if (!trialActive) {
      maybeRegressPhase3NoGoOnEligiblePokeEnd();
    }
    if (trainingPhase == PHASE_2_HOLD_CUE &&
        currentActiveSide == side &&
        !trialActive &&
        !mh.isTimeoutActive() &&
        duration < (TRIAL_START_HOLD_MS * 1000ULL)) {
      startTaskTimeout("Phase2EarlyWithdraw");
    }
    *eligibleFlag = false;
  }

  resolveTrialOnPokeEnd(side, pokeEndUnix, pokeStartUnix);
}

void handlePokeEndEdges() {
  bool leftNow = mh.leftPokeActive();
  bool rightNow = mh.rightPokeActive();

  bool leftEnded = (!leftNow && prevLeftPokeActive);
  bool rightEnded = (!rightNow && prevRightPokeActive);

  handleOnePokeEnd(SIDE_LEFT, leftEnded, leftPokeStartUnix, &leftPokeEligible);
  handleOnePokeEnd(SIDE_RIGHT, rightEnded, rightPokeStartUnix, &rightPokeEligible);

  prevLeftPokeActive = leftNow;
  prevRightPokeActive = rightNow;
}

void updatePelletAvailability() {
  if (mh.pelletRetrieved()) {
    if (mh.isRunning() && !mh.isTimeoutActive() && isInitiationEligible()) {
      set_trial_available_on();
    }
  }
}

void handleFeedStateTransitions() {
  bool feedNow = mh.isFeedActive();

  if (prevFeedActive && !feedNow) {
    if (mh.isRunning() && !mh.isTimeoutActive()) {
      set_trial_available_on();
    }
  }

  prevFeedActive = feedNow;
}

void handleSessionStart() {
  sessionWasRunning = true;
  trialCount = 0;
  trialActive = false;
  goTimeoutTriggered = false;
  waitingForRelease = false;
  waitingForReleaseSide = -1;

  initializePhaseIfNeeded();
  currentTrialType = pickTrialType();
  currentActiveSide = (currentActiveSide == SIDE_LEFT) ? SIDE_RIGHT : SIDE_LEFT;
  sideSetCount++;
  resetInitiationWindow();
  syncPokeStatesToCurrent();

  logEvent("SIDE_SET",
           get_timestamp_us(), time_us_64(),
           (currentActiveSide == SIDE_LEFT) ? "L" : "R",
           sideSetCount,
           NAN_INT,
           NAN_INT,
           NAN_INT,
           getPokeContext(),
           "StartToggle");

  logEvent("ACK_START",
           get_timestamp_us(), time_us_64(),
           NAN_STR,
           mh.fps(),
           mh.framePeriodUs(),
           NAN_INT,
           NAN_INT,
           getPokeContext(),
           NAN_STR);

  set_trial_available_on();
  logTrialAvailabilityDebug("Start");
  logTaskInfo("Start");
}

void handleSessionStop() {
  sessionWasRunning = false;
  trialActive = false;
  goTimeoutTriggered = false;
  waitingForRelease = false;
  waitingForReleaseSide = -1;
  syncPokeStatesToCurrent();
  set_trial_available_off();

  logEvent("ACK_STOP",
           get_timestamp_us(), time_us_64(),
           NAN_STR,
           mh.frameCounter(),
           NAN_INT,
           NAN_INT,
           NAN_INT,
           getPokeContext(),
           NAN_STR);
}

void setup() {
  if (!rtc.begin()) {
    Serial.println("ERROR_NO_RTC");
    while (1) {}
  }

  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  base_us = time_us_64();
  DateTime now = rtc.now();
  base_unix_us = (uint64_t)now.unixtime() * 1000000ULL;

  randomSeed((uint32_t)time_us_64());
  currentTrialType = pickTrialType();

  mh.setCompatibilitySerialMode(true);
  mh.setTaskContext("Eligible");
  mh.begin();
  mh.setPelletSensorMode(MouseHouse::PELLET_SENSOR_LATCHED_PRESENCE);
  syncPokeStatesToCurrent();
}

void loop() {
  syncLibraryTaskContext();
  mh.update();
  if (mh.timeoutEnded() && mh.isRunning()) {
    // Reset held-poke timing at timeout end so the availability lights can
    // return without the same loop immediately starting a new trial.
    syncPokeStatesToCurrent();
    set_trial_available_on();
    return;
  }

  bool running = mh.isRunning();
  if (!running) {
    if (sessionWasRunning) {
      handleSessionStop();
    } else {
      set_trial_available_off();
    }
    return;
  }

  if (!sessionWasRunning) {
    handleSessionStart();
  }

  handlePokeStartEdges();
  updateTrialWhilePokeHeld();
  handlePokeEndEdges();
  updatePelletAvailability();
  handleFeedStateTransitions();
}
