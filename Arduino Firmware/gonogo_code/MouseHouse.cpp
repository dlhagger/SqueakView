#include "MouseHouse.h"

#include <stdlib.h>
#include <string.h>

#ifndef MPR121_TOUCHTH_0
#define MPR121_TOUCHTH_0 0x41
#define MPR121_RELEASETH_0 0x42
#define MPR121_DEBOUNCE 0x5B
#define MPR121_CONFIG1 0x5C
#define MPR121_CONFIG2 0x5D
#define MPR121_ECR 0x5E
#endif

extern "C" uint64_t time_us_64();

MouseHouse::MouseHouse()
    : cap_(),
      strip_(kNumPixels, kNeoPixelPin, NEO_GRBW + NEO_KHZ800),
      stepper_(kMotorStepsPerRevolution, kAIN1, kAIN2, kBIN1, kBIN2) {}

uint64_t MouseHouse::getTimestampUs() const {
  uint64_t elapsedUs = time_us_64() - baseUs_;
  return baseUnixUs_ + elapsedUs;
}

void MouseHouse::robustShow() {
  strip_.begin();
  strip_.show();
}

void MouseHouse::logEvent(const char* eventType,
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

const char* MouseHouse::getContext() const {
  if (taskContext_[0] != '\0') {
    return taskContext_;
  }
  if (timeoutActive_) {
    return "Timeout";
  }
  if (pelletAvailable_) {
    return "Pellet_Available";
  }
  if (feedActive_) {
    return "Feeding";
  }
  return "Eligible";
}

void MouseHouse::setCompatibilitySerialMode(bool enabled) {
  compatibilitySerialMode_ = enabled;
}

void MouseHouse::setTaskContext(const char* context) {
  if (context == nullptr) {
    taskContext_[0] = '\0';
    return;
  }

  strncpy(taskContext_, context, sizeof(taskContext_) - 1);
  taskContext_[sizeof(taskContext_) - 1] = '\0';
}

void MouseHouse::resetSessionEventCounts() {
  leftPokeEvent_.count = 0;
  rightPokeEvent_.count = 0;
  leftDrinkEvent_.count = 0;
  rightDrinkEvent_.count = 0;
  pelletDeliveryCount_ = 0;
  pelletRetrievalCount_ = 0;
  timeoutCount_ = 0;
}

void MouseHouse::resetSerialCommandBuffer() {
  serialCmdLength_ = 0;
  serialCmdOverflow_ = false;
  serialCmdBuffer_[0] = '\0';
}

void MouseHouse::handleSerialCommand(const char* cmd) {
  if (strncmp(cmd, "START,", 6) == 0) {
    long fps = strtol(cmd + 6, nullptr, 10);
    if (fps > 0 && fps <= 120) {
      startSession((uint32_t)fps);
    }
  } else if (strcmp(cmd, "STOP") == 0) {
    stopSession();
  } else if (strncmp(cmd, "FEED,", 5) == 0) {
    long steps = strtol(cmd + 5, nullptr, 10);
    if (steps > 0) {
      feed((int)steps);
    }
  }
}

void MouseHouse::checkSerialCommands() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();

    if (ch == '\r') continue;

    if (ch == '\n') {
      if (!serialCmdOverflow_ && serialCmdLength_ > 0) {
        serialCmdBuffer_[serialCmdLength_] = '\0';
        handleSerialCommand(serialCmdBuffer_);
      }
      resetSerialCommandBuffer();
      continue;
    }

    if (serialCmdOverflow_) continue;

    if (serialCmdLength_ < (kSerialCmdBufferSize - 1)) {
      serialCmdBuffer_[serialCmdLength_++] = ch;
    } else {
      serialCmdOverflow_ = true;
    }
  }
}

void MouseHouse::setDebounce(uint8_t dt, uint8_t dr) {
  dt = dt & 0x07;
  dr = dr & 0x07;
  uint8_t savedEcr = cap_.readRegister8(MPR121_ECR);
  cap_.writeRegister(MPR121_ECR, 0x00);
  cap_.writeRegister(MPR121_DEBOUNCE, (dr << 4) | dt);
  cap_.writeRegister(MPR121_ECR, savedEcr);
}

void MouseHouse::dumpRegs() {
  uint8_t db = cap_.readRegister8(MPR121_DEBOUNCE);
  Serial.print("Debounce DT=");
  Serial.print(db & 0x07);
  Serial.print(" DR=");
  Serial.println((db >> 4) & 0x07);

  Serial.print("TTH/RTH electrode 1 = ");
  Serial.print(cap_.readRegister8(MPR121_TOUCHTH_0 + 2));
  Serial.print("/");
  Serial.println(cap_.readRegister8(MPR121_RELEASETH_0 + 2));
}

void MouseHouse::configureMPR121() {
  cap_.setAutoconfig(true);
  cap_.setThresholds(2, 0);
  setDebounce(0, 0);
  dumpRegs();
}

void MouseHouse::configureMPR121Silent() {
  cap_.setAutoconfig(true);
  cap_.setThresholds(2, 0);
  setDebounce(0, 0);
}

void MouseHouse::resetMPR121() {
  Serial.println("MPR121_RESET_BEGIN");
  cap_.writeRegister(0x80, 0x63);
  delay(1);
  delay(2);

  if (!cap_.begin(0x5A)) {
    Serial.println("MPR121_RESET_FAIL: begin()");
    return;
  }

  configureMPR121();
  dumpRegs();
  Serial.println("MPR121_RESET_OK");
}

bool MouseHouse::mpr121Faulted() {
  uint8_t ecr = cap_.readRegister8(MPR121_ECR);
  if (ecr == 0x00) return true;

  uint8_t oorL = cap_.readRegister8(0x02);
  uint8_t oorH = cap_.readRegister8(0x03);
  return (oorL | oorH) != 0;
}

void MouseHouse::checkMPR121Health() {
  unsigned long nowMs = millis();
  if (nowMs - lastMpr121Check_ >= kMpr121CheckIntervalMs) {
    lastMpr121Check_ = nowMs;

    if (mpr121Faulted()) {
      Serial.println("MPR121_FAULT_DETECTED");
      resetMPR121();
    }
  }
}

void MouseHouse::updateTimedBinaryEvent(TimedBinaryEvent& event,
                                        bool currentState,
                                        const char* startEventType,
                                        const char* endEventType,
                                        const char* side,
                                        uint64_t nowUnix,
                                        uint64_t nowUs) {
  if (currentState && !event.previousState) {
    event.startTime = nowUnix;
    event.count++;

    logEvent(startEventType,
             nowUnix, nowUs,
             side,
             event.count,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }

  if (!currentState && event.previousState) {
    event.endTime = nowUnix;
    uint64_t duration = event.endTime - event.startTime;
    if (event.count > 0 && duration > 0) {
      event.justEnded = true;

      logEvent(endEventType,
               event.endTime, nowUs,
               side,
               event.count,
               duration,
               kNotApplicable,
               kNotApplicable,
               getContext(),
               kNanString);
    }
  }

  event.previousState = currentState;
}

void MouseHouse::primeTimedBinaryEvent(TimedBinaryEvent& event,
                                       bool currentState,
                                       uint64_t nowUnix) {
  event.previousState = currentState;
  event.startTime = currentState ? nowUnix : 0;
  event.endTime = 0;
  event.justEnded = false;
}

void MouseHouse::syncBehavioralStateToSensors(uint64_t nowUnix) {
  uint16_t touched = cap_.touched();
  bool pelletNow = feedPelletSensorTriggered();

  primeTimedBinaryEvent(leftPokeEvent_, digitalRead(kLeftPokePin) == LOW, nowUnix);
  primeTimedBinaryEvent(rightPokeEvent_, digitalRead(kRightPokePin) == LOW, nowUnix);
  primeTimedBinaryEvent(leftDrinkEvent_, (touched & (1 << 1)) != 0, nowUnix);
  primeTimedBinaryEvent(rightDrinkEvent_, (touched & (1 << 2)) != 0, nowUnix);

  prevPelletState_ = pelletNow;
  pelletJustRetrieved_ = false;
  pelletRetrievalTime_ = 0;
  pelletRetrievalLatency_ = 0;
  pelletArrivalTime_ = 0;
  wellCheckActive_ = false;
  wellCheckStartTime_ = 0;
  wellCheckEndTime_ = 0;
  ignoreNextPelletSensorClear_ = false;

  if (pelletSensorMode_ == PELLET_SENSOR_LATCHED_PRESENCE) {
    pelletAvailable_ = pelletNow;
  } else {
    pelletAvailable_ = false;
  }
}

void MouseHouse::updateLeftPoke() {
  bool currentState = (digitalRead(kLeftPokePin) == LOW);
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  updateTimedBinaryEvent(leftPokeEvent_, currentState,
                         "POKE_START", "POKE_END", "L",
                         nowUnix, nowUs);
}

void MouseHouse::updateRightPoke() {
  bool currentState = (digitalRead(kRightPokePin) == LOW);
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  updateTimedBinaryEvent(rightPokeEvent_, currentState,
                         "POKE_START", "POKE_END", "R",
                         nowUnix, nowUs);
}

void MouseHouse::updatePelletWell() {
  bool pelletNow = (digitalRead(kPelletSensorPin) == LOW);
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  if (feedActive_) {
    // While the feeder is running, the pellet beam belongs to feed
    // confirmation only. Do not generate well checks during this window.
    wellCheckActive_ = false;
    prevPelletState_ = pelletNow;
    return;
  }

  if (pelletSensorMode_ == PELLET_SENSOR_TRANSIENT_DELIVERY) {
    if (!pelletNow && prevPelletState_ && ignoreNextPelletSensorClear_) {
      ignoreNextPelletSensorClear_ = false;
    }

    wellCheckActive_ = false;
    prevPelletState_ = pelletNow;
    return;
  }

  if (pelletNow && !prevPelletState_) {
    if (!pelletAvailable_) {
      wellCheckStartTime_ = nowUnix;
      wellCheckCount_++;
      wellCheckActive_ = true;

      logEvent("WELL_CHECK_START",
               nowUnix, nowUs,
               kNanString,
               wellCheckCount_,
               kNotApplicable, kNotApplicable, kNotApplicable,
               getContext(),
               kNanString);
    }
  }

  if (!pelletNow && prevPelletState_) {
    if (wellCheckActive_) {
      wellCheckEndTime_ = nowUnix;
      uint64_t duration = wellCheckEndTime_ - wellCheckStartTime_;
      wellCheckActive_ = false;

      logEvent("WELL_CHECK_END",
               wellCheckEndTime_, nowUs,
               kNanString,
               wellCheckCount_,
               duration,
               kNotApplicable,
               kNotApplicable,
               getContext(),
               kNanString);
    } else if (pelletAvailable_) {
      recordPelletRetrieved(nowUnix, nowUs, kNanString);
    }
  }

  // In latched-presence mode, the beam staying clear means the pellet is gone.
  // Recover here as well so a missed falling edge cannot leave pelletAvailable_
  // stuck true indefinitely.
  if (pelletAvailable_ && !pelletNow) {
    recordPelletRetrieved(nowUnix, nowUs, "Sensor clear");
  }

  prevPelletState_ = pelletNow;
}

void MouseHouse::updateLeftDrink() {
  bool currentState = cap_.touched() & (1 << 1);
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  updateTimedBinaryEvent(leftDrinkEvent_, currentState,
                         "DRINK_START", "DRINK_END", "LD",
                         nowUnix, nowUs);
}

void MouseHouse::updateRightDrink() {
  bool currentState = cap_.touched() & (1 << 2);
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  updateTimedBinaryEvent(rightDrinkEvent_, currentState,
                         "DRINK_START", "DRINK_END", "RD",
                         nowUnix, nowUs);
}

void MouseHouse::pollBehavioralSensors() {
  updateLeftPoke();
  updateRightPoke();
  updatePelletWell();
  updateLeftDrink();
  updateRightDrink();
}

void MouseHouse::disableFeederOutputs() {
  digitalWrite(kAIN1, LOW);
  digitalWrite(kAIN2, LOW);
  digitalWrite(kBIN1, LOW);
  digitalWrite(kBIN2, LOW);
}

bool MouseHouse::feedPelletSensorTriggered() const {
  return digitalRead(kPelletSensorPin) == LOW;
}

void MouseHouse::logFeedPelletArrival(uint64_t nowUs, const char* reason) {
  logEvent("PELLET_ARRIVAL",
           getTimestampUs(), nowUs,
           kNanString,
           pelletDeliveryCount_,
           kNotApplicable, kNotApplicable, kNotApplicable,
           getContext(),
           reason);
}

void MouseHouse::recordPelletRetrieved(uint64_t nowUnix,
                                       uint64_t nowUs,
                                       const char* reason) {
  if (!pelletAvailable_) {
    return;
  }

  pelletRetrievalTime_ = nowUnix;
  pelletRetrievalCount_++;

  uint64_t duration = kNotApplicable;
  uint64_t latency = kNotApplicable;
  if (pelletArrivalTime_ > 0) {
    duration = pelletRetrievalTime_ - pelletArrivalTime_;
    latency = duration;
  }

  pelletRetrievalLatency_ = latency;
  pelletJustRetrieved_ = true;
  const char* context = getContext();
  pelletAvailable_ = false;
  ignoreNextPelletSensorClear_ = false;

  logEvent("PELLET_RETRIEVAL",
           pelletRetrievalTime_, nowUs,
           kNanString,
           pelletRetrievalCount_,
           duration,
           latency,
           kNotApplicable,
           context,
           reason);

  pelletArrivalTime_ = 0;
}
void MouseHouse::startFeedRun(int steps, uint64_t nowUs) {
  feedActive_ = true;
  feedRequestedSteps_ = steps;
  feedStepDirection_ = feedPreferredStepDirection_;
  feedStepDelayUs_ = kFeedStepDelayUs;
  feedStepsLeft_ = feedRequestedSteps_;
  feedCurrentStep_ = 0;
  feedNextStepTime_ = nowUs;
  feedRetryCount_ = 0;

  feedStartTime_ = nowUs;
  feedStartCount_++;

  uint64_t nowUnix = getTimestampUs();

  logEvent("FEED_START",
           nowUnix, nowUs,
           kNanString,
           feedStartCount_,
           kNotApplicable,
           kNotApplicable,
           steps,
           getContext(),
           kNanString);
  digitalWrite(kMotorEnablePin, HIGH);
}

void MouseHouse::queueFeedRetry(uint64_t nowUs) {
  feedRetryCount_++;
  feedPreferredStepDirection_ = -feedPreferredStepDirection_;
  feedStepDirection_ = feedPreferredStepDirection_;
  feedStepDelayUs_ = kFeedRetryStepDelayUs;
  feedStepsLeft_ = feedRequestedSteps_;
  feedCurrentStep_ = 0;
  feedNextStepTime_ = nowUs + feedStepDelayUs_;

  logEvent("FEED_RETRY",
           getTimestampUs(), nowUs,
           kNanString,
           feedRetryCount_,
           kNotApplicable,
           kNotApplicable,
           feedRequestedSteps_,
           getContext(),
           "Pellet not detected, flipping retry direction");
}

void MouseHouse::logFeedJam(uint64_t nowUs) {
  logEvent("FEED_JAM",
           getTimestampUs(), nowUs,
           kNanString,
           feedStopCount_ + 1,
           kNotApplicable,
           kNotApplicable,
           kNotApplicable,
           getContext(),
           "Pellet did not trigger sensor");
}

void MouseHouse::advanceFeedMotorStep(uint64_t nowUs) {
  stepper_.step(feedStepDirection_);

  feedCurrentStep_++;
  feedStepsLeft_--;
  feedNextStepTime_ = nowUs + feedStepDelayUs_;
}

void MouseHouse::handleFeedPelletArrival(uint64_t nowUs,
                                         const char* arrivalReason,
                                         const char* stopReason) {
  pelletArrivalTime_ = getTimestampUs();
  pelletDeliveryCount_++;
  pelletAvailable_ = (pelletSensorMode_ == PELLET_SENSOR_LATCHED_PRESENCE);
  ignoreNextPelletSensorClear_ = (pelletSensorMode_ == PELLET_SENSOR_TRANSIENT_DELIVERY);
  logFeedPelletArrival(nowUs, arrivalReason);
  feedStop(stopReason);
  feedRetryCount_ = 0;
}

void MouseHouse::handleFeedPassComplete(uint64_t nowUs) {
  if (feedPelletSensorTriggered()) {
    handleFeedPelletArrival(nowUs, "Pellet after full pass", "Feed complete");
  } else if (feedRetryCount_ < kMaxFeedRetries) {
    queueFeedRetry(nowUs);
  } else {
    logFeedJam(nowUs);
    feedStop("Feeder Jammed: User Check");
    feedRetryCount_ = 0;
  }
}

void MouseHouse::feedStop(const char* reason) {
  feedActive_ = false;
  disableFeederOutputs();
  digitalWrite(kMotorEnablePin, LOW);
  indicators_.refreshNeeded = indicators_.mainStrip.isOn
                              || indicators_.rightPoke.isOn
                              || indicators_.leftPoke.isOn;

  feedStopCount_++;

  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();
  uint64_t duration = (feedStartTime_ > 0) ? (nowUs - feedStartTime_) : kNotApplicable;

  logEvent("FEED_STOP",
           nowUnix, nowUs,
           kNanString,
           feedStopCount_,
           duration,
           kNotApplicable,
           kNotApplicable,
           getContext(),
           reason);

  robustShow();
}

void MouseHouse::feed(int steps) {
  uint64_t nowUs = time_us_64();

  if (steps > 0 && !feedActive_) {
    startFeedRun(steps, nowUs);
  }
}

void MouseHouse::serviceFeed() {
  uint64_t nowUs = time_us_64();

  if (!feedActive_) return;

  if (feedPelletSensorTriggered()) {
    handleFeedPelletArrival(time_us_64(), kNanString, "Pellet detected mid-feed");
    return;
  }

  if (nowUs < feedNextStepTime_) return;

  advanceFeedMotorStep(nowUs);

  if (feedStepsLeft_ <= 0) {
    handleFeedPassComplete(time_us_64());
  }
}

bool MouseHouse::anyIndicatorsOn() const {
  return indicators_.mainStrip.isOn
         || indicators_.rightPoke.isOn
         || indicators_.leftPoke.isOn;
}

void MouseHouse::renderIndicators() {
  uint32_t stripColor = indicators_.mainStrip.isOn ? indicators_.mainStrip.color : 0;
  uint32_t rightPokeColor = indicators_.rightPoke.isOn ? indicators_.rightPoke.color : 0;
  uint32_t leftPokeColor = indicators_.leftPoke.isOn ? indicators_.leftPoke.color : 0;

  for (int i = 0; i < kActivePixels; i++) {
    strip_.setPixelColor(i, stripColor);
  }

  strip_.setPixelColor(kRightPokeLedIndex, rightPokeColor);
  strip_.setPixelColor(kLeftPokeLedIndex, leftPokeColor);
}

void MouseHouse::turnIndicatorOn(IndicatorChannel& channel,
                                 uint32_t colorVal,
                                 const char* eventType) {
  bool wasOn = channel.isOn;
  bool colorChanged = (!channel.isOn || channel.color != colorVal);

  digitalWrite(kMotorEnablePin, HIGH);

  channel.isOn = true;
  channel.color = colorVal;

  if (colorChanged || indicators_.refreshNeeded) {
    renderIndicators();
    robustShow();
  }

  bool suppressEvent = compatibilitySerialMode_
                       && (strcmp(eventType, "LEFT_POKE_LIGHT_ON") == 0
                           || strcmp(eventType, "RIGHT_POKE_LIGHT_ON") == 0);

  if ((!wasOn || colorChanged) && !suppressEvent) {
    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent(eventType,
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             colorVal,
             getContext(),
             kNanString);
  }

  channel.lastColor = colorVal;
  indicators_.refreshNeeded = false;
}

void MouseHouse::turnIndicatorOff(IndicatorChannel& channel,
                                  const char* eventType) {
  if (!channel.isOn) {
    if (!anyIndicatorsOn()) {
      digitalWrite(kMotorEnablePin, LOW);
    }
    indicators_.refreshNeeded = false;
    return;
  }

  channel.isOn = false;
  channel.color = 0;
  renderIndicators();
  robustShow();

  if (!anyIndicatorsOn()) {
    digitalWrite(kMotorEnablePin, LOW);
  }

  indicators_.refreshNeeded = false;

  bool suppressEvent = compatibilitySerialMode_
                       && (strcmp(eventType, "LEFT_POKE_LIGHT_OFF") == 0
                           || strcmp(eventType, "RIGHT_POKE_LIGHT_OFF") == 0);

  if (!suppressEvent) {
    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent(eventType,
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             channel.lastColor,
             getContext(),
             kNanString);
  }
}

void MouseHouse::setMainStrip(uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
  turnIndicatorOn(indicators_.mainStrip,
                  strip_.Color(r, g, b, w),
                  "STRIP_ON");
}

void MouseHouse::clearMainStrip() {
  turnIndicatorOff(indicators_.mainStrip, "STRIP_OFF");
}

void MouseHouse::rightPokeLightOn(uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
  turnIndicatorOn(indicators_.rightPoke,
                  strip_.Color(r, g, b, w),
                  "RIGHT_POKE_LIGHT_ON");
}

void MouseHouse::rightPokeLightOff() {
  turnIndicatorOff(indicators_.rightPoke, "RIGHT_POKE_LIGHT_OFF");
}

void MouseHouse::leftPokeLightOn(uint8_t r, uint8_t g, uint8_t b, uint8_t w) {
  turnIndicatorOn(indicators_.leftPoke,
                  strip_.Color(r, g, b, w),
                  "LEFT_POKE_LIGHT_ON");
}

void MouseHouse::leftPokeLightOff() {
  turnIndicatorOff(indicators_.leftPoke, "LEFT_POKE_LIGHT_OFF");
}

void MouseHouse::flashMainStrip(uint8_t r, uint8_t g, uint8_t b, uint64_t durationMs) {
  uint32_t colorVal = strip_.Color(r, g, b, 0);

  setMainStrip(r, g, b);
  ledFlashActive_ = true;
  ledFlashEndTime_ = time_us_64() + (durationMs * 1000ULL);

  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();

  logEvent("STRIP_FLASH_START",
           nowUnix, nowUs,
           kNanString,
           kNotApplicable,
           durationMs * 1000ULL,
           kNotApplicable,
           colorVal,
           getContext(),
           kNanString);
}

void MouseHouse::updateLEDFlash() {
  if (ledFlashActive_ && time_us_64() > ledFlashEndTime_) {
    clearMainStrip();
    ledFlashActive_ = false;

    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent("STRIP_FLASH_END",
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             indicators_.mainStrip.lastColor,
             getContext(),
             kNanString);
  }
}

void MouseHouse::playTone(unsigned int freq, uint64_t durationMs) {
  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();
  uint64_t durationUs = durationMs * 1000ULL;

  logEvent("TONE_START",
           nowUnix, nowUs,
           kNanString,
           kNotApplicable,
           durationUs,
           kNotApplicable,
           freq,
           getContext(),
           kNanString);

  tone(kBuzzerPin, freq);
  toneActive_ = true;
  tonePulsed_ = false;
  tonePulseIsOn_ = true;
  tonePulseOnMs_ = 0;
  tonePulseOffMs_ = 0;
  nextToneToggleTime_ = 0;
  toneEndTime_ = nowUs + durationUs;
  lastToneFreq_ = freq;
}

void MouseHouse::playPulsedTone(unsigned int freq,
                                uint64_t durationMs,
                                uint32_t onMs,
                                uint32_t offMs) {
  if (onMs == 0 || offMs == 0 || durationMs == 0) {
    playTone(freq, durationMs);
    return;
  }

  uint64_t nowUnix = getTimestampUs();
  uint64_t nowUs = time_us_64();
  uint64_t durationUs = durationMs * 1000ULL;

  logEvent("TONE_START",
           nowUnix, nowUs,
           kNanString,
           kNotApplicable,
           durationUs,
           kNotApplicable,
           freq,
           getContext(),
           kNanString);

  tone(kBuzzerPin, freq);
  toneActive_ = true;
  tonePulsed_ = true;
  tonePulseIsOn_ = true;
  tonePulseOnMs_ = onMs;
  tonePulseOffMs_ = offMs;
  nextToneToggleTime_ = nowUs + (uint64_t)onMs * 1000ULL;
  toneEndTime_ = nowUs + durationUs;
  lastToneFreq_ = freq;
}

void MouseHouse::updateTone() {
  if (!toneActive_) return;

  uint64_t nowUs = time_us_64();

  if (tonePulsed_ && nowUs < toneEndTime_ && nowUs >= nextToneToggleTime_) {
    if (tonePulseIsOn_) {
      noTone(kBuzzerPin);
      tonePulseIsOn_ = false;
      nextToneToggleTime_ = nowUs + (uint64_t)tonePulseOffMs_ * 1000ULL;
    } else {
      tone(kBuzzerPin, lastToneFreq_);
      tonePulseIsOn_ = true;
      nextToneToggleTime_ = nowUs + (uint64_t)tonePulseOnMs_ * 1000ULL;
    }
  }

  if (nowUs > toneEndTime_) {
    noTone(kBuzzerPin);
    toneActive_ = false;
    tonePulsed_ = false;
    tonePulseIsOn_ = false;
    tonePulseOnMs_ = 0;
    tonePulseOffMs_ = 0;
    nextToneToggleTime_ = 0;

    uint64_t nowUnix = getTimestampUs();

    logEvent("TONE_END",
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             lastToneFreq_,
             getContext(),
             kNanString);
  }
}

void MouseHouse::startClickPattern(int clicks,
                                   unsigned int freq,
                                   uint32_t clickOnMs,
                                   uint32_t clickOffMs) {
  if (clicks <= 0) return;

  clickPatternActive_ = true;
  clickPatternEndPending_ = false;
  clickPatternRemaining_ = clicks;
  nextClickTime_ = time_us_64();
  clickPatternFreq_ = freq;
  clickPatternOnMs_ = clickOnMs;
  clickPatternOffMs_ = clickOffMs;

  logEvent("CLICK_PATTERN_START",
           getTimestampUs(), time_us_64(),
           kNanString,
           clicks,
           kNotApplicable,
           kNotApplicable,
           clickPatternFreq_,
           getContext(),
           kNanString);
}

void MouseHouse::clearClickPattern() {
  clickPatternActive_ = false;
  clickPatternEndPending_ = false;
  clickPatternRemaining_ = 0;
  nextClickTime_ = 0;
}

void MouseHouse::updateClickPattern() {
  if (!clickPatternActive_) return;

  uint64_t nowUs = time_us_64();

  if (clickPatternEndPending_) {
    if (nowUs >= nextClickTime_) {
      clearClickPattern();

      logEvent("CLICK_PATTERN_END",
               getTimestampUs(), nowUs,
               kNanString,
               kNotApplicable,
               kNotApplicable,
               kNotApplicable,
               clickPatternFreq_,
               getContext(),
               kNanString);
    }
    return;
  }

  if (nowUs < nextClickTime_) return;

  if (clickPatternRemaining_ <= 0) {
    clickPatternEndPending_ = true;
    nextClickTime_ = nowUs;
    return;
  }

  playTone(clickPatternFreq_, clickPatternOnMs_);
  clickPatternRemaining_--;
  nextClickTime_ = nowUs + (clickPatternOnMs_ + clickPatternOffMs_) * 1000ULL;

  if (clickPatternRemaining_ == 0) {
    clickPatternEndPending_ = true;
  }
}

void MouseHouse::clearTimeoutState() {
  bool timeoutWasSignaling = timeoutOwnsTone_ || clickPatternActive_ || clickPatternEndPending_;

  timeoutActive_ = false;
  timeoutEndTime_ = 0;
  timeoutJustEnded_ = false;
  timeoutSignalMode_ = TIMEOUT_SILENT;
  timeoutOwnsTone_ = false;

  if (timeoutWasSignaling && toneActive_) {
    noTone(kBuzzerPin);
    toneActive_ = false;

    logEvent("TONE_END",
             getTimestampUs(), time_us_64(),
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             lastToneFreq_,
             getContext(),
             kNanString);
  }

  clearClickPattern();
}

void MouseHouse::startTimeout(uint64_t durationMs,
                              const char* reason,
                              TimeoutSignalMode mode,
                              unsigned int clickFreq,
                              uint32_t clickOnMs,
                              uint32_t clickOffMs,
                              int clickCount) {
  if (timeoutActive_) return;

  timeoutActive_ = true;
  timeoutEndTime_ = time_us_64() + (durationMs * 1000ULL);
  timeoutJustEnded_ = false;
  timeoutCount_++;
  timeoutSignalMode_ = mode;
  timeoutOwnsTone_ = false;

  clearMainStrip();
  rightPokeLightOff();
  leftPokeLightOff();

  if (timeoutSignalMode_ == TIMEOUT_CLICK_PATTERN) {
    startClickPattern(clickCount, clickFreq, clickOnMs, clickOffMs);
  } else if (timeoutSignalMode_ == TIMEOUT_STEADY_TONE) {
    playTone(clickFreq, durationMs);
    timeoutOwnsTone_ = true;
  }

  logEvent("TIMEOUT_START",
           getTimestampUs(), time_us_64(),
           kNanString,
           timeoutCount_,
           durationMs * 1000ULL,
           kNotApplicable,
           kNotApplicable,
           getContext(),
           reason ? reason : kNanString);
}

void MouseHouse::updateTimeout() {
  if (!timeoutActive_) return;

  if (time_us_64() >= timeoutEndTime_) {
    timeoutActive_ = false;
    timeoutEndTime_ = 0;
    timeoutJustEnded_ = true;
    timeoutSignalMode_ = TIMEOUT_SILENT;
    timeoutOwnsTone_ = false;

    logEvent("TIMEOUT_END",
             getTimestampUs(), time_us_64(),
             kNanString,
             timeoutCount_,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }
}

void MouseHouse::houseLightOn() {
  if (!houseLightIsOn_) {
    digitalWrite(kHouseLightPin, LOW);
    houseLightIsOn_ = true;

    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent("HOUSELIGHT_ON",
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             1,
             getContext(),
             kNanString);
  }
}

void MouseHouse::houseLightOff() {
  if (houseLightIsOn_) {
    digitalWrite(kHouseLightPin, HIGH);
    houseLightIsOn_ = false;

    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent("HOUSELIGHT_OFF",
             nowUnix, nowUs,
             kNanString,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             0,
             getContext(),
             kNanString);
  }
}

void MouseHouse::updateHouseLight() {
  DateTime now = rtc_.now();
  int hour = now.hour();

  if (hour >= 5 && hour < 17) {
    houseLightOn();
  } else {
    houseLightOff();
  }
}

void MouseHouse::cameraTTLs(uint64_t nowUs) {
  if (!pulsing_ && nowUs >= nextFrameStart_) {
    digitalWrite(kPulsePin, HIGH);
    pulsing_ = true;
    pulseStartTime_ = nowUs;
    frameCounter_++;
    nextFrameStart_ += framePeriodUs_;

    uint64_t nowUnix = getTimestampUs();
    uint64_t preciseNowUs = time_us_64();

    logEvent("CAMERA_HIGH",
             nowUnix, preciseNowUs,
             kNanString,
             frameCounter_,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }

  if (pulsing_ && (nowUs - pulseStartTime_ >= kPulseWidthUs)) {
    digitalWrite(kPulsePin, LOW);
    pulsing_ = false;

    uint64_t nowUnix = getTimestampUs();
    uint64_t preciseNowUs = time_us_64();

    logEvent("CAMERA_LOW",
             nowUnix, preciseNowUs,
             kNanString,
             frameCounter_,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }
}

void MouseHouse::updateBackgroundServices() {
  checkSerialCommands();
  serviceFeed();
  checkMPR121Health();
  updateTone();
  updateClickPattern();
  updateTimeout();
  updateLEDFlash();
  updateHouseLight();
}

bool MouseHouse::consumeEventFlag(TimedBinaryEvent& event) {
  bool triggered = event.justEnded;
  event.justEnded = false;
  return triggered;
}

bool MouseHouse::consumeFlag(bool& flag) {
  bool triggered = flag;
  flag = false;
  return triggered;
}

void MouseHouse::begin() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!rtc_.begin()) {
    Serial.println("ERROR_NO_RTC");
    while (1)
      ;
  }

  if (rtc_.lostPower()) {
    Serial.println("RTC_LOST_POWER");
    rtc_.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  baseUs_ = time_us_64();
  DateTime now = rtc_.now();
  baseUnixUs_ = (uint64_t)now.unixtime() * 1000000ULL;

  pinMode(kPulsePin, OUTPUT);
  digitalWrite(kPulsePin, LOW);

  pinMode(kLeftPokePin, INPUT_PULLUP);
  pinMode(kRightPokePin, INPUT_PULLUP);
  pinMode(kPelletSensorPin, INPUT_PULLUP);

  pinMode(kAIN1, OUTPUT);
  pinMode(kAIN2, OUTPUT);
  pinMode(kBIN1, OUTPUT);
  pinMode(kBIN2, OUTPUT);
  pinMode(kMotorEnablePin, OUTPUT);
  digitalWrite(kMotorEnablePin, LOW);
  stepper_.setSpeed(kFeedMotorRpm);

  framePeriodUs_ = 1000000ULL / fps_;
  nextFrameStart_ = time_us_64() + framePeriodUs_;

  if (!cap_.begin(0x5A)) {
    Serial.println("ERROR_NO_MPR121");
    while (1)
      ;
  }

  configureMPR121();
  dumpRegs();

  pinMode(kBuzzerPin, OUTPUT);
  digitalWrite(kBuzzerPin, LOW);

  strip_.begin();
  strip_.clear();
  strip_.show();

  pinMode(kHouseLightPin, OUTPUT);
  digitalWrite(kHouseLightPin, HIGH);

  Serial.print("SYSTEM_START,");
  Serial.print(getTimestampUs());
  Serial.print(",");
  Serial.println(time_us_64());
  if (!compatibilitySerialMode_) {
    Serial.print("MOUSEHOUSE_BUILD,");
    Serial.print(__DATE__);
    Serial.print(",");
    Serial.println(__TIME__);
  }
}

void MouseHouse::startSession(uint32_t fps) {
  if (fps == 0 || fps > 120) return;

  configureMPR121Silent();
  clearTimeoutState();

  fps_ = fps;
  running_ = true;
  frameCounter_ = 0;
  pulsing_ = false;
  resetSessionEventCounts();
  syncBehavioralStateToSensors(getTimestampUs());

  framePeriodUs_ = 1000000ULL / fps_;
  nextFrameStart_ = time_us_64() + framePeriodUs_;

  if (!compatibilitySerialMode_) {
    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent("ACK_START",
             nowUnix, nowUs,
             kNanString,
             fps_,
             framePeriodUs_,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }
}

void MouseHouse::stopSession() {
  if (feedActive_) {
    feedStop("StopCommand");
    feedRetryCount_ = 0;
  }

  running_ = false;
  digitalWrite(kPulsePin, LOW);
  pulsing_ = false;
  clearTimeoutState();

  if (!compatibilitySerialMode_) {
    uint64_t nowUnix = getTimestampUs();
    uint64_t nowUs = time_us_64();

    logEvent("ACK_STOP",
             nowUnix, nowUs,
             kNanString,
             frameCounter_,
             kNotApplicable,
             kNotApplicable,
             kNotApplicable,
             getContext(),
             kNanString);
  }
}

void MouseHouse::update() {
  updateBackgroundServices();

  if (running_) {
    cameraTTLs(time_us_64());
  }

  if (running_ || sensorPollingWhileStopped_) {
    pollBehavioralSensors();
  }
}

bool MouseHouse::isRunning() const {
  return running_;
}

uint32_t MouseHouse::fps() const {
  return fps_;
}

uint64_t MouseHouse::framePeriodUs() const {
  return framePeriodUs_;
}

uint64_t MouseHouse::frameCounter() const {
  return frameCounter_;
}

bool MouseHouse::isTimeoutActive() const {
  return timeoutActive_;
}

bool MouseHouse::timeoutEnded() {
  return consumeFlag(timeoutJustEnded_);
}

bool MouseHouse::leftPokeEnded() {
  return consumeEventFlag(leftPokeEvent_);
}

bool MouseHouse::rightPokeEnded() {
  return consumeEventFlag(rightPokeEvent_);
}

bool MouseHouse::leftDrinkEnded() {
  return consumeEventFlag(leftDrinkEvent_);
}

bool MouseHouse::rightDrinkEnded() {
  return consumeEventFlag(rightDrinkEvent_);
}

bool MouseHouse::pelletRetrieved() {
  return consumeFlag(pelletJustRetrieved_);
}

void MouseHouse::markPelletRetrieved() {
  recordPelletRetrieved(getTimestampUs(), time_us_64(), "Manual");
}

bool MouseHouse::leftPokeActive() const {
  return leftPokeEvent_.previousState;
}

bool MouseHouse::rightPokeActive() const {
  return rightPokeEvent_.previousState;
}

bool MouseHouse::leftDrinkActive() const {
  return leftDrinkEvent_.previousState;
}

bool MouseHouse::rightDrinkActive() const {
  return rightDrinkEvent_.previousState;
}

bool MouseHouse::pelletSensorBlocked() const {
  return prevPelletState_;
}

bool MouseHouse::isPelletAvailable() const {
  return pelletAvailable_;
}

bool MouseHouse::isFeedActive() const {
  return feedActive_;
}

void MouseHouse::setPelletSensorMode(PelletSensorMode mode) {
  pelletSensorMode_ = mode;
  syncBehavioralStateToSensors(getTimestampUs());
}

MouseHouse::PelletSensorMode MouseHouse::pelletSensorMode() const {
  return pelletSensorMode_;
}

void MouseHouse::setSensorPollingWhileStopped(bool enabled) {
  sensorPollingWhileStopped_ = enabled;
}

bool MouseHouse::sensorPollingWhileStopped() const {
  return sensorPollingWhileStopped_;
}
