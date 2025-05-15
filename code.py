import time
import pigpio
import threading


import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net import samples_loc
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net.engine import HotwordDetector
from eff_word_net.engine import MultiHotwordDetector




base_model = Resnet50_Arc_loss()

expelliarmous_hw = HotwordDetector(
    hotword="expelliarmous",
    model = base_model,
    reference_file="voice_files/expelliarmous_ref.json",
    threshold=0.7,
    relaxation_time=2
)

crucio_hw = HotwordDetector(
        hotword="crucio",
        model=base_model,
        reference_file="voice_files/crucio_ref.json",
        threshold=0.7,
        relaxation_time=2,
        #verbose=True
)


stupify_hw = HotwordDetector(
    hotword="stupify",
    model=base_model,
    reference_file="voice_files/stupify_ref.json",
    threshold=0.7,
    relaxation_time=2,
    #verbose=True
)

multi_hotword_detector = MultiHotwordDetector(
    [expelliarmous_hw, crucio_hw, stupify_hw],
    model=base_model,
    continuous=True,
)

mic_stream = SimpleMicStream(window_length_secs=1.5, sliding_window_secs=0.75)
mic_stream.start_stream()





TX_GPIO = 18  # IR transmitter GPIO
RX_GPIO = 23  # IR receiver GPIO

# Command-to-LED mapping
LED_PINS = {
    0b00: 4,   # Red
    0b01: 12,  # Blue
    0b10: 16   # Yellow
}

# CLI command labels
COMMAND_LABELS = {
    "red": 0b00,
    "blue": 0b01,
    "yellow": 0b10
}

# Initialize pigpio
pi = pigpio.pi()
if not pi.connected:
    print("Could not connect to pigpio daemon.")
    exit()

# State variables
valid_detect = True
shooting_disabled = False
last_tick = None
dead = False
health = 100
pulse_durations = []
pulse_times = []

# --- Precise IR sender using pigpio waveforms ---
def send_ir_command(command_bits):
    print(f"Sending command: {bin(command_bits)}")

    pulses = []

    for i in range(2):
        bit = (command_bits >> (1 - i)) & 1
        duration = 15000 if bit == 1 else 8000
        cycles = int(38000 * duration / 1e6)
        on = int(1e6 / 38000 / 2)
        off = on

        for _ in range(cycles):
            pulses.append(pigpio.pulse(1 << TX_GPIO, 0, on))
            pulses.append(pigpio.pulse(0, 1 << TX_GPIO, off))

        pulses.append(pigpio.pulse(0, 0, 5000))  # 5ms inter-bit gap

    pi.wave_clear()
    pi.wave_add_generic(pulses)
    wid = pi.wave_create()
    if wid >= 0:
        pi.wave_send_once(wid)
        while pi.wave_tx_busy():
            time.sleep(0.001)
        pi.wave_delete(wid)

# --- LED control ---
def turn_off_led(pin):
    time.sleep(2.5)
    pi.write(pin, 0)
    global valid_detect
    valid_detect = True
    print("LED off")
    
def turn_off_led_exp(pin):
    time.sleep(2.5)
    global valid_detect
    valid_detect = True
    time.sleep(2.5)
    print("LED off")
    pi.write(pin, 0)
    
def turn_shooting_back_on(dur):
    time.sleep(dur)
    global shooting_disabled
    shooting_disabled = False
    print("Shooting Enabled")

# --- IR receiver logic ---
def rx_callback(gpio, level, tick):
    global last_tick, pulse_durations, valid_detect, pulse_times

    if not valid_detect:
        return

    current_time = time.time()

    # Reset if pulses are too far apart
    if pulse_times and (current_time - pulse_times[-1] > 0.1):
        pulse_durations.clear()
        pulse_times.clear()
        last_tick = None

    print(f"Edge detected: level={level}, tick={tick}")  # Debug line

    if level == 0:  # Falling edge = pulse starts
        last_tick = tick

    elif level == 1 and last_tick is not None:  # Rising edge = pulse ends
        duration = tick - last_tick
        pulse_durations.append(duration)
        pulse_times.append(current_time)
        last_tick = None

        if len(pulse_durations) == 2:
            print(f"Durations received: {pulse_durations}")
            command = 0
            for duration in pulse_durations:
                if 5000 <= duration <= 9000:
                    bit = 0
                elif 12000 <= duration <= 16000:
                    bit = 1
                else:
                    print(f"Ignoring invalid pulse duration: {duration}")
                    pulse_durations.clear()
                    pulse_times.clear()
                    return

                command = (command << 1) | bit

            if command not in LED_PINS:
                print(f"Unknown command: {bin(command)}")
                pulse_durations.clear()
                pulse_times.clear()
                return

            led_pin = LED_PINS[command]
            global shooting_disabled
            global dead
            global health
            if led_pin == 12:
                print("Shooting Disabled")
                shooting_disabled = True
                health -= 5
                if health <= 0:
                    valid_detect = False
                    pi.write(4, 1)
                    pi.write(12, 1)
                    pi.write(16, 1)
                    dead = True
                    return
                else:
                    threading.Thread(target=turn_shooting_back_on, args=(5,)).start()
                    threading.Thread(target=turn_off_led_exp, args=(led_pin,)).start()


            if led_pin == 4:
                print("Shooting Disabled")
                shooting_disabled = True
                health -= 20
                if health <= 0:
                    valid_detect = False
                    pi.write(4, 1)
                    pi.write(12, 1)
                    pi.write(16, 1)
                    dead = True
                    return
                else:
                    threading.Thread(target=turn_shooting_back_on, args=(1,)).start()

            if led_pin == 16:
                health -= 25
                if health <= 0:
                    valid_detect = False
                    shooting_disabled = True
                    pi.write(4, 1)
                    pi.write(12, 1)
                    pi.write(16, 1)
                    dead = True
                    return

            print(f"IR decoded: {bin(command)} -> LED {led_pin}")
            pi.write(led_pin, 1)
            if (led_pin != 12):
                threading.Thread(target=turn_off_led, args=(led_pin,)).start()
            valid_detect = False
            print("HEALTH: " + str(health))
            
            

            pulse_durations.clear()
            pulse_times.clear()



# --- GPIO Setup ---
pi.set_mode(TX_GPIO, pigpio.OUTPUT)
pi.set_mode(RX_GPIO, pigpio.INPUT)
pi.set_pull_up_down(RX_GPIO, pigpio.PUD_UP)

for pin in LED_PINS.values():
    pi.set_mode(pin, pigpio.OUTPUT)
    pi.write(pin, 0)

pi.callback(RX_GPIO, pigpio.EITHER_EDGE, rx_callback)

def on_expelliarmous():
    send_ir_command(COMMAND_LABELS['blue'])
    print('expell')


def on_crucio():
    send_ir_command(COMMAND_LABELS['yellow'])
    print('crucio')


def on_stupify():
    send_ir_command(COMMAND_LABELS['red'])
    print('stupify')


# --- Interactive CLI ---
try:
    while True:
        frame = mic_stream.getFrame()
        result = multi_hotword_detector.findBestMatch(frame)
        if (dead):
            print("YOU LOSE")
            time.sleep(5)
            break
        if(None not in result):
            word, confidence = result
            print(word, f", Confidence {confidence:0.4f}")
            word = str(word).replace("Hotword: ","").strip()
            
            if (shooting_disabled):
                print("YOU CAN't Shoot")
	    continue
    

            if word == "expelliarmous":
                on_expelliarmous()
            elif word == "crucio":
                on_crucio()
            elif word == "stupify":
                on_stupify()

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    pi.set_PWM_dutycycle(TX_GPIO, 0)
    for pin in LED_PINS.values():
        pi.write(pin, 0)
    pi.stop()


