#include <Arduino.h>
#include <RTClib.h>
#include <Adafruit_MPR121.h>
#include <Adafruit_NeoPixel.h>

// ----------------------
// Hardware setup
// ----------------------
RTC_DS3231 rtc;
Adafruit_MPR121 cap = Adafruit_MPR121();

const int pulsePin = A0;  // camera sync output
#define LEFT_POKE 6       // IR beam-break for left poke
#define RIGHT_POKE 5      // IR beam-break for right poke
#define PELLET_SENSOR 0   // IR beam-break for pellet well (D0 / RX pin)

// Stepper motor pins
const int AIN1 = A3;
const int AIN2 = A2;
const int BIN1 = 24;
const int BIN2 = 25;
const int MOTOR_EN = 13;

// ----------------------
// Sync pulse variables
// ----------------------
volatile uint32_t FPS = 30;
const uint64_t pulseWidth_us = 1000;  // 1 ms pulse width

bool pulsing = false;
bool running = false;

uint64_t base_us = 0;
uint64_t base_unix_us = 0;

uint64_t nextFrameStart = 0;
uint64_t frameCounter = 0;
uint64_t pulseStartTime = 0;
uint64_t framePeriod_us = 0;

// ----------------------
// Poke variables
// ----------------------
bool prevLeftPoke = false;
uint64_t pokeLeftStart = 0;
uint64_t pokeLeftEnd = 0;
bool leftPokeJustEnded = false;
unsigned long leftPokeCount = 0;

bool prevRightPoke = false;
uint64_t pokeRightStart = 0;
uint64_t pokeRightEnd = 0;
bool rightPokeJustEnded = false;
unsigned long rightPokeCount = 0;

// ----------------------
// Drink variables
// ----------------------
bool prevLeftDrink = false;
uint64_t drinkLeftStart = 0;
uint64_t drinkLeftEnd = 0;
unsigned long leftDrinkCount = 0;
bool leftDrinkJustEnded = false;

bool prevRightDrink = false;
uint64_t drinkRightStart = 0;
uint64_t drinkRightEnd = 0;
unsigned long rightDrinkCount = 0;
bool rightDrinkJustEnded = false;

// ----------------------
// Pellet + Well variables
// ----------------------

// Beam-break state
bool prevPelletState = false;  // last sensor state (LOW = beam broken)

// Pellet retrieval tracking
uint64_t pelletArrivalTime = 0;          // unix timestamp when pellet became available
uint64_t pelletRetrievalTime = 0;        // unix timestamp when pellet retrieved
uint64_t pelletRetrievalLatency = 0;     // µs between arrival and retrieval
unsigned long pelletRetrievalCount = 0;  // total retrievals so far
bool pelletJustRetrieved = false;        // flag for contingent logic
bool pelletAvailable = false;            // true if pellet is sitting in well

// Well check tracking (empty checks)
uint64_t wellCheckStartTime = 0;   // unix timestamp when empty check started
uint64_t wellCheckEndTime = 0;     // unix timestamp when empty check ended
unsigned long wellCheckCount = 0;  // total empty checks so far
bool wellCheckActive = false;      // true if currently in a check

// Feeder bookkeeping
uint64_t feedStartTime = 0;        // µs timestamp when feed() begins
unsigned long feedStartCount = 0;  // number of feed starts
unsigned long feedStopCount = 0;   // number of feed stops

// ----------------------
// Buzzer variables
// ----------------------
#define BUZZER_PIN 1

bool toneActive = false;
uint64_t toneEndTime = 0;

// ----------------------
// NeoPixel setup
// ----------------------
#define NEOPIXEL_PIN A1  // your chosen pin for the strip
#define NUM_PIXELS 10    // total number of NeoPixels

// Using GRBW here since that's what your old code used
Adafruit_NeoPixel strip(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRBW + NEO_KHZ800);

bool ledFlashActive = false;
uint64_t ledFlashEndTime = 0;

void robustShow() {
  strip.begin();  // re-init hardware state (needed on your setup)
  strip.show();   // push current buffer to strip
}

// ----------------------
// houslight setup
// ----------------------
#define HOUSELIGHT_PIN 9
bool houseLightIsOn = false;  // track current state

// ----------------------
// Feeder variables
// ----------------------
bool feedActive = false;
int feedStepsLeft = 0;
int feedCurrentStep = 0;
uint64_t feedNextStepTime = 0;
const uint64_t FEED_STEP_DELAY_US = 5000ULL;

const int MAX_FEED_RETRIES = 1;   // how many extra passes after first
int feedRetryCount = 0;           // moved here, was already declared below
int feedRequestedSteps = 0;       // remember how many steps to run per pass

const int motorSequence[4][4] = {
  { 1, 0, 1, 0 },
  { 0, 1, 1, 0 },
  { 0, 1, 0, 1 },
  { 1, 0, 0, 1 }
};

// ----------------------
// Helpers
// ----------------------
extern "C" uint64_t time_us_64();

uint64_t get_timestamp_us() {
  uint64_t elapsed_us = time_us_64() - base_us;
  return base_unix_us + elapsed_us;
}

#define NAN_STR "nan"
#define NAN_COUNT 0
#define NAN_NUM 0
#define NAN_INT 69420  // use this as "not applicable" for numeric fields



// Central logger for all 10-column events
void logEvent(const char* eventType,
              uint64_t unixTime,    // DS3231 time in µs
              uint64_t rp2040Time,  // RP2040 uptime in µs
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

const char* getPokeContext() {
  if (digitalRead(PELLET_SENSOR) == LOW) {
    return "Pellet_Available";
  } else if (feedActive) {
    return "Feeding";
  }
  return "Eligible";
}

// ----------------------
// MPR121 functions + helpers
// ----------------------
#ifndef MPR121_TOUCHTH_0
#define MPR121_TOUCHTH_0 0x41
#define MPR121_RELEASETH_0 0x42
#define MPR121_DEBOUNCE 0x5B
#define MPR121_CONFIG1 0x5C
#define MPR121_CONFIG2 0x5D
#define MPR121_ECR 0x5E
#endif

// --- MPR121 watchdog (1 min) ---
unsigned long lastMPR121Check = 0;
const unsigned long MPR121_CHECK_INTERVAL_MS = 60000UL;  // 1 minute

void setDebounce(uint8_t dt, uint8_t dr) {
  dt = dt & 0x07;
  dr = dr & 0x07;
  uint8_t savedECR = cap.readRegister8(MPR121_ECR);  // save run mode
  cap.writeRegister(MPR121_ECR, 0x00);               // stop
  cap.writeRegister(MPR121_DEBOUNCE, (dr << 4) | dt);
  cap.writeRegister(MPR121_ECR, savedECR);  // restore run mode
}

void dump_regs() {
  uint8_t db = cap.readRegister8(MPR121_DEBOUNCE);
  Serial.print("Debounce DT=");
  Serial.print(db & 0x07);
  Serial.print(" DR=");
  Serial.println((db >> 4) & 0x07);

  Serial.print("TTH/RTH electrode 1 = ");
  Serial.print(cap.readRegister8(MPR121_TOUCHTH_0 + 2));
  Serial.print("/");
  Serial.println(cap.readRegister8(MPR121_RELEASETH_0 + 2));
}

void configureMPR121() {
  cap.setAutoconfig(true);

  cap.setThresholds(2, 0);  // touch, release
  setDebounce(0, 0);        // DT, DR

  dump_regs();  // optional, but handy for debugging
}

void configureMPR121_silent() {
  // Enable autoconfig
  cap.setAutoconfig(true);

  // Sensitive threshold profile
  cap.setThresholds(2, 0);  // touch=2, release=0
  setDebounce(0, 0);        // DT=0, DR=0

  // No printing, no dump_regs()
}


// Datasheet: soft reset register
static inline void mpr121SoftReset() {
  cap.writeRegister(0x80, 0x63);  // soft reset
  delay(1);
}

// Considered "faulted" if device stopped (ECR=0) or OOR flags set
static inline bool mpr121Faulted() {
  uint8_t ecr = cap.readRegister8(MPR121_ECR);
  if (ecr == 0x00) return true;  // in STOP mode

  // OOR (Out-of-Range) status registers: 0x02 (L), 0x03 (H)
  uint8_t oorL = cap.readRegister8(0x02);
  uint8_t oorH = cap.readRegister8(0x03);
  if ((oorL | oorH) != 0) return true;  // any electrode out-of-range

  return false;
}

// Full re-init sequence (safe during long recordings)
void resetMPR121() {
  Serial.println("MPR121_RESET_BEGIN");

  // Stop, soft reset, re-begin, then reapply your tuning
  mpr121SoftReset();
  delay(2);

  // Re-init device (same address & bus)
  if (!cap.begin(0x5A)) {
    Serial.println("MPR121_RESET_FAIL: begin()");
    return;
  }

  // Re-apply your settings
  configureMPR121();

  // Optional: quick sanity dump
  dump_regs();

  Serial.println("MPR121_RESET_OK");
}

// Call this from loop(); runs every minute
void checkMPR121Health() {
  unsigned long nowMs = millis();
  if (nowMs - lastMPR121Check >= MPR121_CHECK_INTERVAL_MS) {
    lastMPR121Check = nowMs;

    if (mpr121Faulted()) {
      Serial.println("MPR121_FAULT_DETECTED");
      resetMPR121();
    }
  }
}

// ----------------------
// Feeder functions
// ----------------------
void feedStop(const char* reason = "Stopped") {
  feedActive = false;
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);
  digitalWrite(MOTOR_EN, LOW);

  feedStopCount++;

  uint64_t now_unix_feedstop = get_timestamp_us();
  uint64_t now_us_feedstop = time_us_64();
  uint64_t duration_feed = (feedStartTime > 0) ? (now_us_feedstop - feedStartTime) : NAN_INT;

  logEvent("FEED_STOP",
           now_unix_feedstop, now_us_feedstop,
           NAN_STR,        // Side
           feedStopCount,  // Count
           duration_feed,  // Duration (µs)
           NAN_INT,        // Latency
           NAN_INT,        // Value
           getPokeContext(),
           reason);

  robustShow();
}

void feed(int steps = 0) {
  uint64_t now_us = time_us_64();

  // --- Feed initialization (only when called with steps > 0 and idle) ---
  if (steps > 0 && !feedActive) {
    feedActive        = true;
    feedRequestedSteps = steps;        // remember requested steps
    feedStepsLeft     = steps;
    feedCurrentStep   = 0;
    feedNextStepTime  = now_us;
    feedRetryCount    = 0;             // new run → reset retries

    feedStartTime = now_us;            // mark motor start
    feedStartCount++;

    uint64_t now_unix = get_timestamp_us();

    // Log FEED_START
    logEvent("FEED_START",
             now_unix, now_us,
             NAN_STR,         // Side
             feedStartCount,  // Count
             NAN_INT,         // Duration unknown yet
             NAN_INT,         // Latency
             steps,           // Value = step count
             getPokeContext(),
             NAN_STR);

    digitalWrite(MOTOR_EN, HIGH);
  }

  // If not active, nothing to do
  if (!feedActive) return;

  // --- Check pellet sensor mid-feed ---
  if (digitalRead(PELLET_SENSOR) == LOW) {
    pelletAvailable = true;

    logEvent("PELLET_ARRIVAL",
             get_timestamp_us(), time_us_64(),
             NAN_STR,
             pelletRetrievalCount + 1,  // next retrieval index
             NAN_INT, NAN_INT, NAN_INT,
             getPokeContext(),
             NAN_STR);

    feedStop("Pellet detected mid-feed");
    feedRetryCount = 0;
    return;
  }

  // --- Stepper motor timing ---
  if (now_us < feedNextStepTime) return;

  int idx = feedCurrentStep % 4;
  digitalWrite(AIN1, motorSequence[idx][0]);
  digitalWrite(AIN2, motorSequence[idx][1]);
  digitalWrite(BIN1, motorSequence[idx][2]);
  digitalWrite(BIN2, motorSequence[idx][3]);

  feedCurrentStep++;
  feedStepsLeft--;
  feedNextStepTime = now_us + FEED_STEP_DELAY_US;

  // --- End of this pass (all steps done) ---
  if (feedStepsLeft <= 0) {
    now_us = time_us_64();  // refresh timestamp

    if (digitalRead(PELLET_SENSOR) == LOW) {
      // Beam broke → pellet delivered
      pelletAvailable = true;

      logEvent("PELLET_ARRIVAL",
               get_timestamp_us(), now_us,
               NAN_STR,
               pelletRetrievalCount + 1,
               NAN_INT, NAN_INT, NAN_INT,
               getPokeContext(),
               "Pellet after full pass");

      feedStop("Feed complete");
      feedRetryCount = 0;
    } else {
      // No pellet detected → decide retry vs jam
      if (feedRetryCount < MAX_FEED_RETRIES) {
        // Re-arm for another full pass
        feedRetryCount++;
        feedStepsLeft    = feedRequestedSteps;
        feedCurrentStep  = 0;
        feedNextStepTime = now_us + FEED_STEP_DELAY_US;

        logEvent("FEED_RETRY",
                 get_timestamp_us(), now_us,
                 NAN_STR,
                 feedRetryCount,       // Count = retry index
                 NAN_INT, NAN_INT,
                 feedRequestedSteps,   // Value = steps per pass
                 getPokeContext(),
                 "Pellet not detected, retrying");
      } else {
        // True jam
        logEvent("FEED_JAM",
                 get_timestamp_us(), now_us,
                 NAN_STR,
                 feedStopCount + 1,  // upcoming stop count
                 NAN_INT, NAN_INT, NAN_INT,
                 getPokeContext(),
                 "Pellet did not trigger sensor");

        feedStop("Feeder jammed");
        feedRetryCount = 0;
      }
    }
  }
}

// ----------------------
// Serial command handler
// ----------------------
void checkSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("START")) {
      int commaIndex = cmd.indexOf(',');
      if (commaIndex > 0) {
        int fpsVal = cmd.substring(commaIndex + 1).toInt();
        if (fpsVal > 0 && fpsVal <= 120) {
          configureMPR121_silent();

          FPS = fpsVal;
          running = true;
          frameCounter = 0;
          pulsing = false;

          // Reset counts
          leftPokeCount = 0;
          rightPokeCount = 0;
          leftDrinkCount = 0;
          rightDrinkCount = 0;
          pelletRetrievalCount = 0;

          framePeriod_us = 1000000ULL / FPS;
          nextFrameStart = time_us_64() + framePeriod_us;

          uint64_t now_unix = get_timestamp_us();
          uint64_t now_us = time_us_64();

          // Unified ACK_START log
          logEvent("ACK_START",
                   now_unix, now_us,
                   NAN_STR,           // Side
                   FPS,               // Count = FPS
                   framePeriod_us,    // Duration = frame period in µs
                   NAN_INT,           // Latency = N/A
                   NAN_INT,           // Value = N/A
                   getPokeContext(),  // Context
                   NAN_STR);          // Reason
        }
      }
    } else if (cmd.equals("STOP")) {
      running = false;
      digitalWrite(pulsePin, LOW);
      pulsing = false;

      uint64_t now_unix = get_timestamp_us();
      uint64_t now_us = time_us_64();

      // Unified ACK_STOP log
      logEvent("ACK_STOP",
               now_unix, now_us,
               NAN_STR,           // Side
               frameCounter,      // Count = how many frames ran
               NAN_INT,           // Duration
               NAN_INT,           // Latency
               NAN_INT,           // Value
               getPokeContext(),  // Context
               NAN_STR);          // Reason
    } else if (cmd.startsWith("FEED")) {
      int commaIndex = cmd.indexOf(',');
      if (commaIndex > 0) {
        int steps = cmd.substring(commaIndex + 1).toInt();
        feed(steps);  // FEED_START/STOP events already logged inside feed()
      }
    }
  }
}

// ----------------------
// camera ttl gen
// ----------------------
void camera_ttls(uint64_t now) {
  // Frame rising edge
  if (!pulsing && now >= nextFrameStart) {
    digitalWrite(pulsePin, HIGH);
    pulsing = true;
    pulseStartTime = now;
    frameCounter++;

    nextFrameStart += framePeriod_us;

    uint64_t now_unix_cam = get_timestamp_us();
    uint64_t now_us_cam = time_us_64();

    logEvent("CAMERA_HIGH",
             now_unix_cam, now_us_cam,
             NAN_STR,           // Side
             frameCounter,      // Count = frame number
             NAN_INT,           // Duration
             NAN_INT,           // Latency
             NAN_INT,           // Value
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }

  // Frame falling edge
  if (pulsing && (now - pulseStartTime >= pulseWidth_us)) {
    digitalWrite(pulsePin, LOW);
    pulsing = false;

    uint64_t now_unix_cam = get_timestamp_us();
    uint64_t now_us_cam = time_us_64();

    logEvent("CAMERA_LOW",
             now_unix_cam, now_us_cam,
             NAN_STR,           // Side
             frameCounter,      // Count = frame number
             NAN_INT,           // Duration
             NAN_INT,           // Latency
             NAN_INT,           // Value
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }
}

// ----------------------
// Left poke sensing
// ----------------------
void left_poke() {
  bool pokeLeftNow = (digitalRead(LEFT_POKE) == LOW);
  uint64_t now_unix_poke = get_timestamp_us();
  uint64_t now_us_poke = time_us_64();

  // POKE_START
  if (pokeLeftNow && !prevLeftPoke) {
    pokeLeftStart = now_unix_poke;
    leftPokeCount++;

    logEvent("POKE_START",
             now_unix_poke, now_us_poke,
             "L",               // Side
             leftPokeCount,     // Count
             NAN_INT,           // Duration N/A
             NAN_INT,           // Latency N/A
             NAN_INT,           // Value N/A
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }

  // POKE_END
  if (!pokeLeftNow && prevLeftPoke) {
    pokeLeftEnd = now_unix_poke;
    uint64_t duration = pokeLeftEnd - pokeLeftStart;
    if (duration > 0) {
      leftPokeJustEnded = true;

      logEvent("POKE_END",
               pokeLeftEnd, now_us_poke,
               "L",               // Side
               leftPokeCount,     // Count
               duration,          // Duration in µs
               NAN_INT,           // Latency N/A
               NAN_INT,           // Value N/A
               getPokeContext(),  // Context
               NAN_STR);          // Reason
    }
  }

  prevLeftPoke = pokeLeftNow;
}

// ----------------------
// Right poke sensing
// ----------------------
void right_poke() {
  bool pokeRightNow = (digitalRead(RIGHT_POKE) == LOW);
  uint64_t now_unix_poke = get_timestamp_us();
  uint64_t now_us_poke = time_us_64();

  // POKE_START
  if (pokeRightNow && !prevRightPoke) {
    pokeRightStart = now_unix_poke;
    rightPokeCount++;

    logEvent("POKE_START",
             now_unix_poke, now_us_poke,
             "R",               // Side
             rightPokeCount,    // Count
             NAN_INT,           // Duration N/A
             NAN_INT,           // Latency N/A
             NAN_INT,           // Value N/A
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }

  // POKE_END
  if (!pokeRightNow && prevRightPoke) {
    pokeRightEnd = now_unix_poke;
    uint64_t duration = pokeRightEnd - pokeRightStart;
    if (duration > 0) {
      rightPokeJustEnded = true;

      logEvent("POKE_END",
               pokeRightEnd, now_us_poke,
               "R",               // Side
               rightPokeCount,    // Count
               duration,          // Duration in µs
               NAN_INT,           // Latency N/A
               NAN_INT,           // Value N/A
               getPokeContext(),  // Context
               NAN_STR);          // Reason
    }
  }

  prevRightPoke = pokeRightNow;
}

// ----------------------
// Pellet well sensing
// ----------------------
void pellet_well() {
  bool pelletNow = (digitalRead(PELLET_SENSOR) == LOW);  // beam broken
  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  // Rising edge: beam just broken
  if (pelletNow && !prevPelletState) {
    if (!pelletAvailable) {
      // Empty well check start
      wellCheckStartTime = now_unix;
      wellCheckCount++;
      wellCheckActive = true;

      logEvent("WELL_CHECK_START",
               now_unix, now_us,
               NAN_STR,
               wellCheckCount,  // Count
               NAN_INT, NAN_INT, NAN_INT,
               getPokeContext(),
               NAN_STR);
    }
    // If pelletAvailable == true, we don’t log arrival here anymore —
    // arrival is handled in feed() when the sensor first breaks.
    // But we *do* store the time so we can calculate retrieval duration/latency.
    else {
      pelletArrivalTime = now_unix;
    }
  }

  // Falling edge: beam just cleared
  if (!pelletNow && prevPelletState) {
    if (pelletAvailable && pelletArrivalTime > 0) {
      // Pellet retrieved
      pelletRetrievalTime = now_unix;
      pelletRetrievalCount++;
      pelletRetrievalLatency = pelletRetrievalTime - pelletArrivalTime;  // latency from arrival to retrieval
      uint64_t duration = pelletRetrievalTime - pelletArrivalTime;       // how long beam was broken

      pelletJustRetrieved = true;
      pelletAvailable = false;

      logEvent("PELLET_RETRIEVAL",
               pelletRetrievalTime, now_us,
               NAN_STR,
               pelletRetrievalCount,
               duration,                // Duration = beam break length
               pelletRetrievalLatency,  // Latency = arrival→retrieval
               NAN_INT,                 // Value = N/A
               getPokeContext(),
               NAN_STR);

      pelletArrivalTime = 0;  // reset after logging
    } else if (wellCheckActive) {
      // End of an empty well check
      wellCheckEndTime = now_unix;
      uint64_t duration = wellCheckEndTime - wellCheckStartTime;
      wellCheckActive = false;

      logEvent("WELL_CHECK_END",
               wellCheckEndTime, now_us,
               NAN_STR,
               wellCheckCount,
               duration,  // Duration = beam break length
               NAN_INT,   // Latency = not applicable
               NAN_INT,   // Value = not applicable
               getPokeContext(),
               NAN_STR);
    }
  }

  prevPelletState = pelletNow;
}

// ----------------------
// Lick sensing
// ----------------------
void left_drink() {
  // Read electrode 1 (bit 1 in cap.touched())
  bool drinkNowLeft = cap.touched() & (1 << 1);

  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  // Drink START
  if (drinkNowLeft && !prevLeftDrink) {
    drinkLeftStart = now_unix;
    leftDrinkCount++;

    logEvent("DRINK_START",
             now_unix, now_us,
             "LD",              // Side
             leftDrinkCount,    // Count
             NAN_INT,           // Duration = not known yet
             NAN_INT,           // Latency
             NAN_INT,           // Value
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }

  // Drink END
  if (!drinkNowLeft && prevLeftDrink) {
    drinkLeftEnd = now_unix;
    uint64_t duration = drinkLeftEnd - drinkLeftStart;
    if (duration > 0) {
      leftDrinkJustEnded = true;  // 🔑 contingent logic flag

      logEvent("DRINK_END",
               drinkLeftEnd, now_us,
               "LD",              // Side
               leftDrinkCount,    // Count
               duration,          // Duration = µs beam broken
               NAN_INT,           // Latency
               NAN_INT,           // Value
               getPokeContext(),  // Context
               NAN_STR);          // Reason
    }
  }

  prevLeftDrink = drinkNowLeft;
}


void right_drink() {
  // Read electrode 2 (bit 2 in cap.touched())
  bool drinkNowRight = cap.touched() & (1 << 2);

  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  // Drink START
  if (drinkNowRight && !prevRightDrink) {
    drinkRightStart = now_unix;
    rightDrinkCount++;

    logEvent("DRINK_START",
             now_unix, now_us,
             "RD",              // Side = Right Drink
             rightDrinkCount,   // Count
             NAN_INT,           // Duration N/A
             NAN_INT,           // Latency N/A
             NAN_INT,           // Value N/A
             getPokeContext(),  // Context
             NAN_STR);          // Reason
  }

  // Drink END
  if (!drinkNowRight && prevRightDrink) {
    drinkRightEnd = now_unix;
    uint64_t duration = drinkRightEnd - drinkRightStart;
    if (duration > 0) {
      rightDrinkJustEnded = true;  // 🔑 contingent logic flag

      logEvent("DRINK_END",
               drinkRightEnd, now_us,
               "RD",              // Side = Right Drink
               rightDrinkCount,   // Count
               duration,          // Duration in µs
               NAN_INT,           // Latency N/A
               NAN_INT,           // Value N/A
               getPokeContext(),  // Context
               NAN_STR);          // Reason
    }
  }

  prevRightDrink = drinkNowRight;
}

// ----------------------
// Buzzer fxns
// ----------------------
unsigned int lastToneFreq = 0;  // store frequency of the last tone

void playTone(unsigned int freq, uint64_t duration_ms) {
  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();
  uint64_t duration_us = duration_ms * 1000ULL;

  // Unified log (10 columns)
  logEvent("TONE_START",
           now_unix, now_us,
           NAN_STR,      // Side
           NAN_INT,      // Count
           duration_us,  // Duration in µs
           NAN_INT,      // Latency
           freq,         // Value = frequency
           getPokeContext(),
           NAN_STR);  // Reason

  tone(BUZZER_PIN, freq);
  toneActive = true;
  toneEndTime = now_us + duration_us;  // µs resolution
  lastToneFreq = freq;                 // 🔑 save frequency for TONE_END
}

void updateTone() {
  if (toneActive && time_us_64() > toneEndTime) {
    noTone(BUZZER_PIN);
    toneActive = false;

    uint64_t now_unix = get_timestamp_us();
    uint64_t now_us = time_us_64();

    // Unified log (10 columns)
    logEvent("TONE_END",
             now_unix, now_us,
             NAN_STR,       // Side
             NAN_INT,       // Count
             NAN_INT,       // Duration N/A
             NAN_INT,       // Latency N/A
             lastToneFreq,  // Value = frequency (from playTone)
             getPokeContext(),
             NAN_STR);  // Reason
  }
}

// ----------------------
// NeoPixel helpers
// ----------------------
unsigned long lastStripColor = 0;  // store color as packed int for END logs

void light_strip_on(uint8_t r, uint8_t g, uint8_t b, uint8_t w = 0) {
  digitalWrite(MOTOR_EN, HIGH);  // power rail for NeoPixels

  uint32_t colorVal = strip.Color(r, g, b, w);
  lastStripColor = colorVal;  // save last color

  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, colorVal);
  }
  robustShow();

  // Log event
  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  logEvent("STRIP_ON",
           now_unix, now_us,
           NAN_STR,   // Side
           NAN_INT,   // Count
           NAN_INT,   // Duration
           NAN_INT,   // Latency
           colorVal,  // Value = packed color
           getPokeContext(),
           NAN_STR);
}

void light_strip_off() {
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, 0);
  }
  robustShow();
  digitalWrite(MOTOR_EN, LOW);

  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  logEvent("STRIP_OFF",
           now_unix, now_us,
           NAN_STR,
           NAN_INT,
           NAN_INT,
           NAN_INT,
           lastStripColor,  // Value = color it was showing
           getPokeContext(),
           NAN_STR);
}

void flash_strip(uint8_t r, uint8_t g, uint8_t b, uint64_t duration_ms) {
  uint32_t colorVal = strip.Color(r, g, b, 0);
  lastStripColor = colorVal;

  light_strip_on(r, g, b);  // turn LEDs on
  ledFlashActive = true;
  ledFlashEndTime = time_us_64() + (duration_ms * 1000ULL);  // µs resolution

  uint64_t now_unix = get_timestamp_us();
  uint64_t now_us = time_us_64();

  logEvent("STRIP_FLASH_START",
           now_unix, now_us,
           NAN_STR,
           NAN_INT,
           duration_ms * 1000ULL,  // Duration in µs
           NAN_INT,
           colorVal,  // Value = packed color
           getPokeContext(),
           NAN_STR);
}

void updateLEDFlash() {
  if (ledFlashActive && time_us_64() > ledFlashEndTime) {
    light_strip_off();
    ledFlashActive = false;

    uint64_t now_unix = get_timestamp_us();
    uint64_t now_us = time_us_64();

    logEvent("STRIP_FLASH_END",
             now_unix, now_us,
             NAN_STR,
             NAN_INT,
             NAN_INT,
             NAN_INT,
             lastStripColor,  // Value = color that was flashing
             getPokeContext(),
             NAN_STR);
  }
}

// ----------------------
// houselight fxns
// ----------------------
void houseLightOn() {
  if (!houseLightIsOn) {                // only act if it's currently OFF
    digitalWrite(HOUSELIGHT_PIN, LOW);  // active-low ON
    houseLightIsOn = true;

    uint64_t now_unix = get_timestamp_us();
    uint64_t now_us = time_us_64();

    logEvent("HOUSELIGHT_ON",
             now_unix, now_us,
             NAN_STR,  // Side
             NAN_INT,  // Count
             NAN_INT,  // Duration
             NAN_INT,  // Latency
             1,        // Value = ON
             getPokeContext(),
             NAN_STR);  // Reason
  }
}

void houseLightOff() {
  if (houseLightIsOn) {                  // only act if it's currently ON
    digitalWrite(HOUSELIGHT_PIN, HIGH);  // active-low OFF
    houseLightIsOn = false;

    uint64_t now_unix = get_timestamp_us();
    uint64_t now_us = time_us_64();

    logEvent("HOUSELIGHT_OFF",
             now_unix, now_us,
             NAN_STR,
             NAN_INT,
             NAN_INT,
             NAN_INT,
             0,  // Value = OFF
             getPokeContext(),
             NAN_STR);
  }
}

void updateHouseLight() {
  DateTime now = rtc.now();
  int hour = now.hour();

  // Turn ON between 05:00 AM and 05:00 PM
  if (hour >= 5 && hour < 17) {
    houseLightOn();
  } else {
    houseLightOff();
  }
}

// ----------------------
// Setup
// ----------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // Init RTC
  if (!rtc.begin()) {
    Serial.println("ERROR_NO_RTC");
    while (1)
      ;
  }

  if (rtc.lostPower()) {
    Serial.println("RTC_LOST_POWER");
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  // Capture base references
  base_us = time_us_64();
  DateTime now = rtc.now();
  base_unix_us = (uint64_t)now.unixtime() * 1000000ULL;

  pinMode(pulsePin, OUTPUT);
  digitalWrite(pulsePin, LOW);

  pinMode(LEFT_POKE, INPUT_PULLUP);
  pinMode(RIGHT_POKE, INPUT_PULLUP);
  pinMode(PELLET_SENSOR, INPUT_PULLUP);

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(MOTOR_EN, OUTPUT);
  digitalWrite(MOTOR_EN, LOW);

  framePeriod_us = 1000000ULL / FPS;
  nextFrameStart = time_us_64() + framePeriod_us;

  if (!cap.begin(0x5A)) {  // default address
    Serial.println("ERROR_NO_MPR121");
    while (1)
      ;
  }

  configureMPR121();

  dump_regs();

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // --- Initialize NeoPixels ---
  strip.begin();  // init strip
  strip.clear();  // set all pixels to 'off' in RAM
  strip.show();   // push off state to hardware

  pinMode(HOUSELIGHT_PIN, OUTPUT);
  digitalWrite(HOUSELIGHT_PIN, HIGH);  // active-low OFF at startup

  Serial.print("SYSTEM_START,");
  Serial.print(get_timestamp_us());
  Serial.print(",");
  Serial.println(time_us_64());
}


// ----------------------
// Main loop
// ----------------------
void loop() {
  checkSerialCommands();
  checkMPR121Health();
  updateTone();
  updateLEDFlash();
  updateHouseLight();

  uint64_t now = time_us_64();

  // --- Camera sync pulse logic ---
  if (running) {

    // --- camera ttls ---
    camera_ttls(now);

    // --- Feeder stepping ---
    feed();  // you should always check the feeder first in the loop for reasons of software engineering and flag setting!!!!

    // --- Sensors ---
    left_poke();
    right_poke();
    pellet_well();
    left_drink();
    right_drink();

    // --- Contingent logic ---
    // Handle events that just ended this loop iteration

    // Pokes
    if (leftPokeJustEnded) {
      // Example: reward after left poke
      feed(1024);
      leftPokeJustEnded = false;  // reset flag
    }

    if (rightPokeJustEnded) {
      // Play 2 kHz tone for 500 ms
      // playTone(2000, 500);
      // flash_strip(255, 0, 0, 2000);  // flash red for 2 s
      rightPokeJustEnded = false;    // reset flag
    }

    // Drinks
    if (leftDrinkJustEnded) {
      leftDrinkJustEnded = false;  // reset flag
    }

    if (rightDrinkJustEnded) {
      rightDrinkJustEnded = false;  // reset flag
    }

    // Pellet retrieval
    if (pelletJustRetrieved) {
      pelletJustRetrieved = false;  // reset flag
    }
  }
}
