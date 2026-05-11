#ifndef MOUSEHOUSE_H
#define MOUSEHOUSE_H

#include <Arduino.h>
#include <RTClib.h>
#include <Adafruit_MPR121.h>
#include <Adafruit_NeoPixel.h>
#include <Stepper.h>

class MouseHouse {
public:
  static constexpr int kDefaultFeedSteps = 300;

  enum PelletSensorMode {
    PELLET_SENSOR_LATCHED_PRESENCE,
    PELLET_SENSOR_TRANSIENT_DELIVERY
  };

  enum TimeoutSignalMode {
    TIMEOUT_SILENT,
    TIMEOUT_CLICK_PATTERN,
    TIMEOUT_STEADY_TONE
  };

  MouseHouse();

  void begin();
  void update();

  void startSession(uint32_t fps);
  void stopSession();
  bool isRunning() const;
  uint32_t fps() const;
  uint64_t framePeriodUs() const;
  uint64_t frameCounter() const;
  void setCompatibilitySerialMode(bool enabled);
  void setTaskContext(const char* context);

  void feed(int steps = kDefaultFeedSteps);
  void playTone(unsigned int freq, uint64_t durationMs);
  void playPulsedTone(unsigned int freq,
                      uint64_t durationMs,
                      uint32_t onMs,
                      uint32_t offMs);
  void startTimeout(uint64_t durationMs,
                    const char* reason,
                    TimeoutSignalMode mode = TIMEOUT_CLICK_PATTERN,
                    unsigned int clickFreq = 2000,
                    uint32_t clickOnMs = 50,
                    uint32_t clickOffMs = 50,
                    int clickCount = 3);
  bool isTimeoutActive() const;
  bool timeoutEnded();

  bool leftPokeEnded();
  bool rightPokeEnded();
  bool leftDrinkEnded();
  bool rightDrinkEnded();
  bool pelletRetrieved();
  bool leftPokeActive() const;
  bool rightPokeActive() const;
  bool leftDrinkActive() const;
  bool rightDrinkActive() const;
  bool pelletSensorBlocked() const;
  void markPelletRetrieved();

  bool isPelletAvailable() const;
  bool isFeedActive() const;
  void setPelletSensorMode(PelletSensorMode mode);
  PelletSensorMode pelletSensorMode() const;
  void setSensorPollingWhileStopped(bool enabled);
  bool sensorPollingWhileStopped() const;

  void setMainStrip(uint8_t r, uint8_t g, uint8_t b, uint8_t w = 0);
  void clearMainStrip();
  void flashMainStrip(uint8_t r, uint8_t g, uint8_t b, uint64_t durationMs);
  void rightPokeLightOn(uint8_t r, uint8_t g, uint8_t b, uint8_t w = 0);
  void rightPokeLightOff();
  void leftPokeLightOn(uint8_t r, uint8_t g, uint8_t b, uint8_t w = 0);
  void leftPokeLightOff();

private:
  struct TimedBinaryEvent {
    bool previousState = false;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
    bool justEnded = false;
    unsigned long count = 0;
  };

  struct IndicatorChannel {
    bool isOn = false;
    uint32_t color = 0;
    uint32_t lastColor = 0;
  };

  struct IndicatorState {
    bool refreshNeeded = false;
    IndicatorChannel mainStrip;
    IndicatorChannel rightPoke;
    IndicatorChannel leftPoke;
  };

  static constexpr int kPulsePin = A0;
  static constexpr int kLeftPokePin = 6;
  static constexpr int kRightPokePin = 5;
  static constexpr int kPelletSensorPin = 0;

  static constexpr int kAIN1 = A3;
  static constexpr int kAIN2 = A2;
  static constexpr int kBIN1 = 24;
  static constexpr int kBIN2 = 25;
  static constexpr int kMotorEnablePin = 13;

  static constexpr int kBuzzerPin = 1;
  static constexpr int kNeoPixelPin = A1;
  static constexpr int kHouseLightPin = 9;

  static constexpr uint8_t kNumPixels = 10;
  static constexpr uint8_t kActivePixels = 8;
  static constexpr uint8_t kRightPokeLedIndex = 8;
  static constexpr uint8_t kLeftPokeLedIndex = 9;

  static constexpr uint64_t kPulseWidthUs = 1000ULL;
  static constexpr int kMotorStepsPerRevolution = 2038;
  static constexpr long kFeedMotorRpm = 60;
  static constexpr int kFeedStepDirection = -1;
  static constexpr uint64_t kFeedStepDelayUs = 5000ULL;
  static constexpr uint64_t kFeedRetryStepDelayUs = 5000ULL;
  static constexpr int kMaxFeedRetries = 3;
  static constexpr unsigned long kMpr121CheckIntervalMs = 60000UL;
  static constexpr size_t kSerialCmdBufferSize = 32;
  static constexpr long kNotApplicable = 69420;
  static constexpr const char* kNanString = "nan";

  RTC_DS3231 rtc_;
  Adafruit_MPR121 cap_;
  Adafruit_NeoPixel strip_;
  Stepper stepper_;

  volatile uint32_t fps_ = 30;
  bool pulsing_ = false;
  bool running_ = false;
  bool sensorPollingWhileStopped_ = false;

  uint64_t baseUs_ = 0;
  uint64_t baseUnixUs_ = 0;
  uint64_t nextFrameStart_ = 0;
  uint64_t frameCounter_ = 0;
  uint64_t pulseStartTime_ = 0;
  uint64_t framePeriodUs_ = 0;

  TimedBinaryEvent leftPokeEvent_;
  TimedBinaryEvent rightPokeEvent_;
  TimedBinaryEvent leftDrinkEvent_;
  TimedBinaryEvent rightDrinkEvent_;

  bool prevPelletState_ = false;
  uint64_t pelletArrivalTime_ = 0;
  uint64_t pelletRetrievalTime_ = 0;
  uint64_t pelletRetrievalLatency_ = 0;
  unsigned long pelletDeliveryCount_ = 0;
  unsigned long pelletRetrievalCount_ = 0;
  bool pelletJustRetrieved_ = false;
  bool pelletAvailable_ = false;
  PelletSensorMode pelletSensorMode_ = PELLET_SENSOR_LATCHED_PRESENCE;
  bool ignoreNextPelletSensorClear_ = false;

  uint64_t wellCheckStartTime_ = 0;
  uint64_t wellCheckEndTime_ = 0;
  unsigned long wellCheckCount_ = 0;
  bool wellCheckActive_ = false;

  uint64_t feedStartTime_ = 0;
  unsigned long feedStartCount_ = 0;
  unsigned long feedStopCount_ = 0;
  bool feedActive_ = false;
  int feedStepsLeft_ = 0;
  int feedCurrentStep_ = 0;
  int feedPreferredStepDirection_ = kFeedStepDirection;
  int feedStepDirection_ = kFeedStepDirection;
  uint64_t feedStepDelayUs_ = kFeedStepDelayUs;
  uint64_t feedNextStepTime_ = 0;
  int feedRetryCount_ = 0;
  int feedRequestedSteps_ = 0;

  bool toneActive_ = false;
  uint64_t toneEndTime_ = 0;
  unsigned int lastToneFreq_ = 0;
  bool tonePulsed_ = false;
  bool tonePulseIsOn_ = false;
  uint32_t tonePulseOnMs_ = 0;
  uint32_t tonePulseOffMs_ = 0;
  uint64_t nextToneToggleTime_ = 0;

  bool timeoutActive_ = false;
  uint64_t timeoutEndTime_ = 0;
  unsigned long timeoutCount_ = 0;
  bool timeoutJustEnded_ = false;
  TimeoutSignalMode timeoutSignalMode_ = TIMEOUT_SILENT;
  bool timeoutOwnsTone_ = false;

  bool clickPatternActive_ = false;
  bool clickPatternEndPending_ = false;
  int clickPatternRemaining_ = 0;
  uint64_t nextClickTime_ = 0;
  unsigned int clickPatternFreq_ = 2000;
  uint32_t clickPatternOnMs_ = 50;
  uint32_t clickPatternOffMs_ = 50;

  bool ledFlashActive_ = false;
  uint64_t ledFlashEndTime_ = 0;
  IndicatorState indicators_;

  bool houseLightIsOn_ = false;

  unsigned long lastMpr121Check_ = 0;

  char serialCmdBuffer_[kSerialCmdBufferSize];
  size_t serialCmdLength_ = 0;
  bool serialCmdOverflow_ = false;
  bool compatibilitySerialMode_ = false;
  char taskContext_[32] = "";

  uint64_t getTimestampUs() const;
  void robustShow();
  void logEvent(const char* eventType,
                uint64_t unixTime,
                uint64_t rp2040Time,
                const char* side,
                unsigned long count,
                uint64_t duration,
                uint64_t latency,
                long value,
                const char* context,
                const char* reason);
  const char* getContext() const;

  void resetSessionEventCounts();
  void resetSerialCommandBuffer();
  void handleSerialCommand(const char* cmd);
  void checkSerialCommands();

  void setDebounce(uint8_t dt, uint8_t dr);
  void dumpRegs();
  void configureMPR121();
  void configureMPR121Silent();
  void resetMPR121();
  void checkMPR121Health();
  bool mpr121Faulted();

  void updateTimedBinaryEvent(TimedBinaryEvent& event,
                              bool currentState,
                              const char* startEventType,
                              const char* endEventType,
                              const char* side,
                              uint64_t nowUnix,
                              uint64_t nowUs);
  void primeTimedBinaryEvent(TimedBinaryEvent& event,
                             bool currentState,
                             uint64_t nowUnix);
  void syncBehavioralStateToSensors(uint64_t nowUnix);
  void pollBehavioralSensors();
  void updateLeftPoke();
  void updateRightPoke();
  void updatePelletWell();
  void updateLeftDrink();
  void updateRightDrink();

  void disableFeederOutputs();
  bool feedPelletSensorTriggered() const;
  void logFeedPelletArrival(uint64_t nowUs, const char* reason);
  void recordPelletRetrieved(uint64_t nowUnix,
                             uint64_t nowUs,
                             const char* reason);
  void startFeedRun(int steps, uint64_t nowUs);
  void queueFeedRetry(uint64_t nowUs);
  void logFeedJam(uint64_t nowUs);
  void advanceFeedMotorStep(uint64_t nowUs);
  void serviceFeed();
  void handleFeedPelletArrival(uint64_t nowUs,
                               const char* arrivalReason,
                               const char* stopReason);
  void handleFeedPassComplete(uint64_t nowUs);
  void feedStop(const char* reason = "Stopped");

  bool anyIndicatorsOn() const;
  void renderIndicators();
  void turnIndicatorOn(IndicatorChannel& channel,
                       uint32_t colorVal,
                       const char* eventType);
  void turnIndicatorOff(IndicatorChannel& channel,
                        const char* eventType);
  void updateLEDFlash();
  void updateTone();
  void startClickPattern(int clicks,
                         unsigned int freq,
                         uint32_t clickOnMs,
                         uint32_t clickOffMs);
  void clearClickPattern();
  void updateClickPattern();
  void clearTimeoutState();
  void updateTimeout();
  void houseLightOn();
  void houseLightOff();
  void updateHouseLight();

  void cameraTTLs(uint64_t nowUs);
  void updateBackgroundServices();

  bool consumeEventFlag(TimedBinaryEvent& event);
  bool consumeFlag(bool& flag);
};

#endif
