

from concurrent.futures import ThreadPoolExecutor
import threading, time, socket, random, math
import pandas as pd
import numpy as np

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey
from src.utils import relaunch, unpause_game, switch_tab


# konstanta
delta_time = 0.2
saturation = 1
terminal_pitch_target = 60

# initialization
simulation_running = True
flight_data_stack = []

# berapa jumlah thread untku worker
elevator_worker_num = 3
rudder_worker_num = 3

# sinkronisasi antar thread: therad yg mengeksekusi harus menunggu data masuk. 
# saat data masuk thread observer memberi sinyal dengan men-set observation_event
observation_event = threading.Event()

# Mutex/Lock: untuk mengunci sebuah variable supaya komunikasi antar thread mulus
elevator_mutex = threading.Lock()
elevator_threads_communication = {f"Elevator_{i}": threading.Event() for i in range(elevator_worker_num)}
rudder_mutex = threading.Lock()
rudder_threads_communication = {f"Rudder_{i}": threading.Event() for i in range(rudder_worker_num)}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# method2 yang akan dipakai oleh threads
def get_observation():
    print('observation thread started...\n')
    while simulation_running:
        try:
            if sock and not sock._closed:
                d, a = sock.recvfrom(2048) # 2048 maximum bit to receive
                
            values = readPacket(d)          # extract the data received

            latitude = values.get('latitude')
            longitude = values.get('longitude')
            altitude = values.get('agl')
            velocity = values.get('velocity')
            ver_vel = values.get('ver_vel')
    
            roll = values.get('roll')
            roll_rate = values.get('roll_rate')
            pitch = values.get('pitch')
            pitch_rate = values.get('pitch_rate')
            yaw = values.get('yaw') if values.get('yaw') <= 180 else (values.get('yaw') - 360)
            yaw_rate = values.get('yaw_rate')
            
            new_pitch = values.get('new_pitch')
            new_yaw = values.get('new_yaw') # this is new heading or the line of sight for the yaw to target.
            
            is_grounded = values.get('grounded') or values.get('destroyed')
            
            target_latitude = values.get('target_lat')
            target_longitude = values.get('target_long')
                            
            
            flight_data_stack.append([tuple([altitude,
                                                velocity, ver_vel, 
                                                roll, roll_rate, 
                                                pitch, pitch_rate, 
                                                yaw, yaw_rate, 
                                                new_pitch, new_yaw,
                                                latitude, longitude,
                                                target_latitude, target_longitude]), 
                                                is_grounded])
        
            # memberi sinyal ke thread yang lain bahwa data sudah diterima dan bisa diproses
            observation_event.set()
            observation_event.clear()
            
        except socket.timeout:
            # jika terjadi error maka semua data = 0

            print('\n[ERROR]: TimeoutError at get_observation method\n')
            is_grounded = True
            flight_data_stack.append([tuple([0] * 15), is_grounded])
            
            # event set
            observation_event.set()
            observation_event.clear()   
            
    observation_event.clear()
    print('observation thread ended...\n')
    observation_event.set()

# function untuk mengendalikan elevator (pitch)
def elevator_control(pwm):
    thread_name = threading.current_thread().name

    elevator_mutex.acquire()
    elevator_threads_communication[thread_name].set()  # Signal thread communication active
    elevator_mutex.release()


    if pwm > 0:
        sign = 1
        saturated = abs(min(delta_time * saturation, pwm))
        pressKey('s')
        time.sleep(saturated)

    elif pwm < 0:
        sign = -1
        saturated = abs(max(-(delta_time * saturation), pwm))
        pressKey('w')
        time.sleep(saturated)

    else:
        sign = 1
        saturated = 0


    elevator_mutex.acquire()
    elevator_threads_communication[thread_name].clear()  # Signal thread communication complete
    if not any(event.is_set() for event in elevator_threads_communication.values()):
        releaseKey('w')
        releaseKey('s')
    elevator_mutex.release()

    # print(f'\n---_elevator_control---\nthread name: {thread_name}\nsaturated: {saturated}')
            
def elevator_function():
    P = 0.009 # 0.012 # (default value)
    D = 0.01
    with ThreadPoolExecutor(max_workers=elevator_worker_num, 
                            thread_name_prefix="Elevator") as elevator:
        while simulation_running:
            # menunggu sinyal dari observator thread
            observation_event.wait()
            
            data = flight_data_stack[-1][0]
            pitch = data[8]
            pitch_rate = data[9]
            LOS = data[12] # pitch
            
            lt = LOS - terminal_pitch_target
            lp = LOS - pitch
            
            pitch_err = lt + lp
            
            elevator_pwm = pitch_err * P - pitch_rate * D
            
            elevator.submit(elevator_control, elevator_pwm)
    
    print('elevator thread ended')

# function untuk mengarahkan ke target
def calculate_target_heading(current_pos, target_pos):
    assert len(current_pos) == 2
    assert len(target_pos) == 2
    
    current_lat, current_lon, target_lat, target_lon = map(math.radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])
    delta_lon = target_lon - current_lon
    
    x = math.sin(delta_lon) * math.cos(target_lat)
    y = math.cos(current_lat) * math.sin(target_lat) - math.sin(current_lat) * math.cos(target_lat) * math.cos(delta_lon)
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    # bearing = (bearing + 360) % 360  # Normalize to [0, 360)
    if bearing > 180:
        bearing -= 360 # Normalize to [-180, 180]
    
    return bearing

def calculate_heading_error(current_heading, target_heading):
    heading_error = (target_heading - current_heading + 360) % 360
    if heading_error > 180:
        heading_error -= 360  # Normalize to [-180, 180]
        
    return heading_error

def calculate_distance(current_pos, target_pos, radius=1274.2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])

    # Differences in latitude and longitude
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in the specified radius unit
    distance = radius * c
    return distance
    
# function untuk mengendalikan rudder (yaw)
def rudder_control(pwm):
    thread_name = threading.current_thread().name

    rudder_mutex.acquire()
    rudder_threads_communication[thread_name].set()  # Signal thread communication active
    rudder_mutex.release()

    if pwm > 0:
        sign = 1
        saturated = abs(min(delta_time * saturation, pwm))
        pressKey('d')
        time.sleep(saturated)

    elif pwm < 0:
        sign = -1
        saturated = abs(max(-(delta_time * saturation), pwm))
        pressKey('a')
        time.sleep(saturated)

    else:
        sign = 1
        saturated = 0


    rudder_mutex.acquire()
    rudder_threads_communication[thread_name].clear()  # Signal thread communication complete
    if not any(event.is_set() for event in rudder_threads_communication.values()):
        releaseKey('a')
        releaseKey('d')
    rudder_mutex.release()

    # print(f'\n---_rudder_control---\nthread name: {thread_name}\nsaturated: {saturated}\n---_rudder_control---\n')

def rudder_function():
    current_time = None
    prev_time = None
    prev_target_heading = None
    with ThreadPoolExecutor(max_workers=rudder_worker_num, 
                            thread_name_prefix="Rudder") as rudder:
        while simulation_running:
            # menunggu sinyal dari observator thread
            observation_event.wait()
            
            data = flight_data_stack[-1][0]

            latitude = data[14]
            longitude = data[15]
            yaw = data[10]
            current_pos = [latitude, longitude]
            target_pos = [data[-2], data[-1]]
            
            distance_to_target = calculate_distance(current_pos, target_pos)
            
            target_heading = calculate_target_heading(current_pos, target_pos)
            current_time = time.time()
            
            if prev_time is None or prev_target_heading is None:
                prev_target_heading = target_heading
                prev_time = current_time
                LOS_rate = 0.0
            else:
                # lambda dot or lambda rate: delta_theta/delta_time
                delta_heading = target_heading - prev_target_heading
                delta_time = current_time - prev_time + 0.001
                LOS_rate = delta_heading / delta_time
                
                prev_target_heading = target_heading
                prev_time = current_time
                
            # print(f'\ncur_heading: {yaw}\ntarhet_heading: {target_heading}\n')
            
            heading_error = calculate_heading_error(yaw, target_heading)
            FPA_rate = -1 * LOS_rate
            rudder_pwm = heading_error * 0.025 + (FPA_rate + 0.000001) * 0.008
            
            rudder.submit(rudder_control, rudder_pwm)

def save_data_to_csv():
    timestr = time.strftime("%Y%m%d-%H%M%S")
        
    # state_history_values = pd.DataFrame(black_box, 
    #                                     columns=['pos_x', 'pos_y', 'pos_z', 'altitude', 
    #                                                 'velocity', 'ver_vel', 'roll', 'roll_rate',
    #                                                 'pitch', 'pitch_rate', 'yaw', 'yaw_rate',
    #                                                 'longitude', 'latitude'])

    # state_history_filename = f"flight_data/state_history_{timestr}.xlsx"
    # with pd.ExcelWriter(state_history_filename) as writer:
    #     state_history_values.to_excel(writer)            
            
    # action_history_values = pd.DataFrame(action_history, 
    #                                     columns=['aileron_pwm', 'saturated_aileron_pwm'])

    # action_history_filename = f"flight_data/action_history_{timestr}.xlsx"
    # with pd.ExcelWriter(action_history_filename) as writer:
    #     action_history_values.to_excel(writer)
    

# 3 threads: 1. untuk menerima data, 2. untuk mengontrol pitch, 3. untuk mengontrol yaw
observator = threading.Thread(target=get_observation, args=(), name='observator')
pitch_contoller = threading.Thread(target=elevator_function, args=(), name='pitch_contoller')
yaw_controller = threading.Thread(target=rudder_function, args=(), name='yaw_contoller')

def run():
    switch_tab()
    time.sleep(0.5)
    
    # start the threads
    observator.start()
    pitch_contoller.start()
    yaw_controller.start()
    unpause_game()

    grounded = False
    step = 0
    
    while not grounded and step < self.max_step:
        observation_event.wait()
        grounded = flight_data_stack[-1][1]

    simulation_running = False
    print('Simulation ended')
    

# mulai simulasi
run()
