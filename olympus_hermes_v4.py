import asyncio
import time
import base64
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from adafruit_servokit import ServoKit
from picamera2 import Picamera2

# パラメータ設定 (省略せず記述してください)
PAN_CH, TILT_CH = 1, 0
ESC_CH, STR_CH  = 2, 3

ANGLE_STOP, ANGLE_MAX_FWD, ANGLE_MAX_REV = 90, 95, 85
ANGLE_CENTER, ANGLE_LEFT, ANGLE_RIGHT = 86, 61, 111
STEER_GAIN = 1.2

# ==========================================
# 🧠 高度追従エンジン（Jerk制限）
# ==========================================
class SmoothTracker:
    def __init__(self):
        self.curr = [90.0, 90.0]     #現在の角度
        self.target = [90.0, 90.0]   #目標の角度
        self.vel = [0.0, 0.0]        #現在の移動速度
        self.speed_gain = 0.48       #目標値に移動するスピードのゲイン
        self.base_accel = 0.20       #加速度の限界(目標値が近い時)
        self.boost_accel = 0.65      #加速度の限界(目標値が遠い時)
        self.damping = 0.78          #動いた後のブレーキの強さ
        self.deadband = 0.12         #不感帯の設定
        
        
    #アテナ側から届いた顔の角度をサーボの角度に変換
    def update_target(self, yaw, pitch):
        self.target[0] = 90 + yaw
        self.target[1] = 90 + pitch

    #await asyncio.sleep(0.01)(100Hz)で呼び出される度に実行される
    def drive_step(self):
        for i in range(2):
            error = self.target[i] - self.curr[i] #目標値までの残距離計算
            abs_error = abs(error)
            
            # --- ここがポイント：不感帯の代わりに「感度調整」を入れる ---
            if abs_error < 2.0:
                # 誤差が小さい（2度以内）ときは、誤差の大きさに応じて反応を絞る
                # 例：誤差が0.5度なら感度はたったの 0.06 程度になる（超低速）
                dynamic_gain = self.speed_gain * (abs_error / 2.0)
            else:
                dynamic_gain = self.speed_gain
            
            #加速度の決定(残距離が3より上でブースト)
            accel_limit = self.boost_accel if abs(error) > 3.0 else self.base_accel
            
            #速度差の計算 急加速、急減速の防止
            vel_diff = max(-accel_limit, min(accel_limit, (error * dynamic_gain) - self.vel[i]))
            
            #ブレーキの設定。目標値に近づいたら(不感帯の距離に近づいたら)速度を40%に落として静止させる
            self.vel[i] = (self.vel[i] + vel_diff) * self.damping
            
            # 物理的な不感帯は極限まで小さく（あるいは0に）
            if abs_error < 0.05: 
                self.vel[i] = 0
                self.curr[i] = self.target[i] # 完全に近ければ吸着させる
            else:
                self.curr[i] += self.vel[i]
            #サーボ角度の出力(デバッグ用)
            #print(f"Servo Angle -> PAN: {round(self.curr[0], 1)}, TILT: {round(self.curr[1], 1)}", flush=True)
        #サーボへの出力
        kit.servo[PAN_CH].angle = max(0, min(180, round(self.curr[0], 1)))
        kit.servo[TILT_CH].angle = max(0, min(180, round(self.curr[1], 1)))

# ==========================================
# 📡 ネットワークタスク
# ==========================================
# 1. 顔追従UDP受信 & 100Hz駆動ループ
class VisionProtocol(asyncio.DatagramProtocol):
    #UDPデータ(顔の角度)が届いた瞬間に自動で発動するイベントリスナー
    def datagram_received(self, data, addr):
        try:
            # アテナから届いたデータをデコード
            message = data.decode().split(',')
            y = float(message[0])
            p = float(message[1])
            
            # trackerが存在すれば更新
            if tracker:
                tracker.update_target(y, p)
                # デバッグ用：受信したらドットを表示
                # print(".", end="", flush=True) 
        except Exception as e:
            print(f"UDP Decode Error: {e}")
            
#UDP5005番ポートに届くデータは全部VisionProtocolに渡す用に設定
async def vision_control_loop():
    print("📡 UDP Receiver (Standard Mode) Starting on Port 5005...", flush=True)
    
    # 古い方式と同じ socket を使った受信
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 他のプログラムがポートを掴んでいても再利用できるようにする設定
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 5005))
    sock.setblocking(False) # 非同期で動かすためにノンブロッキングに設定
    
    loop = asyncio.get_running_loop()
    
    try:
        while True:
            # データが届くまで待機
            data, addr = await loop.sock_recvfrom(sock, 1024)
            if data and tracker:
                try:
                    # アテナからのデータを解析
                    msg = data.decode().split(',')
                    y = float(msg[0])
                    p = float(msg[1])
                    
                    tracker.update_target(y, p)
                    # デバッグ用：受信したらターミナルに表示（動いたらコメントアウトしてOK）
                    #print(f"Received: {y}, {p}")
                except Exception as e:
                    print(f"UDP Decode Error: {e}")
    except Exception as e:
        print(f"UDP Loop Error: {e}")
    finally:
        sock.close()
        
#100Hz周期でサーボを動かし続けるループ    
async def servo_drive_loop():
    print("🚀 servo_drive_loop STARTED!", flush=True) # これを追加
    try:
        while True:
            tracker.drive_step()
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"servo_drive_error: {e}")

class DriveState:
    def __init__(self):
        self.is_reversing = False

drive_state = DriveState()

# --- グローバル変数 ---
picam2 = None
camera_active = False
tracker = None
kit = None

# --- Lifespan (起動・終了処理) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit,tracker
    # 実行確認用ファイル
    with open("lifespan_check.txt", "w") as f:
        f.write("LIFESPAN STARTED")
        
    print("--- HERMES START ---", flush=True)
    
    try:
        kit = ServoKit(channels=16)
        for ch in [PAN_CH, TILT_CH, ESC_CH, STR_CH]:
            kit.servo[ch].set_pulse_width_range(500, 2500)
        print("✅ ServoKit initialized", flush=True)
    except Exception as e:
        print(f"❌ ServoKit failed: {e}", flush=True)
        kit = None
        
        
    # 1. バックグラウンドタスクを開始
    print("🎬 Starting Background Tasks...", flush=True)
    task1 = asyncio.create_task(vision_control_loop())
    task2 = asyncio.create_task(servo_drive_loop())
    print("🚀 Tasks created. Monitoring UDP and Servo loops.", flush=True)

    # 2. 制御クラスの初期化
    try:
        tracker = SmoothTracker()
        print("✅ SmoothTracker initialized", flush=True)
    except Exception as e:
        print(f"❌ Failed to init SmoothTracker: {e}", flush=True)
        tracker = None
    
    yield # ここでサーバーが動き出す
    
    # 終了処理
    task1.cancel()
    task2.cancel()
    print("--- HERMES SHUTDOWN ---", flush=True)
    if picam2 is not None:
        picam2.stop()
        picam2.close()

# アプリの定義 (lifespanを渡す)
app = FastAPI(lifespan=lifespan)

# --- WebSocket エンドポイント ---

# 2. 走行制御 WebSocket
@app.websocket("/ws/drive")
async def drive_endpoint(websocket: WebSocket):
    #global is_reversing
    await websocket.accept()
    
    #is_reversing = False
    
    try:
        while True:
            data = await websocket.receive_json() #JSONデータが来るまで待機。OSが通知するまでCPU不使用
            #print(f"🕹️ Drive Signal: thr={data.get('throttle')}, steer={data.get('steer')}", flush=True)
            try:
                thr = data['throttle']
                brk = data['brake']
                back = data['isBackMode']
                neut = data['isNeutral']
                steer = data['steer']
            except KeyError as e:
                print(f"Missing key in drive data: {e}")
                continue
            # ステアリング
            if neut:
                kit.servo[STR_CH].angle = ANGLE_CENTER
            else:
                diff = ((float(steer) - 128) / 128 * (ANGLE_RIGHT - ANGLE_CENTER)) * STEER_GAIN
                s_ang = ANGLE_CENTER + diff
                kit.servo[STR_CH].angle = max(min(s_ang, ANGLE_RIGHT), ANGLE_LEFT)
            # ESC
            if back:
                if not drive_state.is_reversing:
                    # 別の非同期タスクとして実行し、メインループを止めない
                    async def reverse_sequence():
                        kit.servo[ESC_CH].angle = ANGLE_MAX_REV
                        await asyncio.sleep(0.1);
                        kit.servo[ESC_CH].angle = ANGLE_STOP
                        await asyncio.sleep(0.1);
                        kit.servo[ESC_CH].angle = ANGLE_MAX_REV
                        drive_state.is_reversing = True
                        asyncio.create_task(reverse_sequence())
                else:
                    kit.servo[ESC_CH].angle = ANGLE_MAX_REV
            elif thr > 0:
                drive_state.is_reversing = False
                kit.servo[ESC_CH].angle = ANGLE_STOP + (thr * (ANGLE_MAX_FWD - ANGLE_STOP))
            elif brk > 0:
                drive_state.is_reversing = False
                kit.servo[ESC_CH].angle = ANGLE_STOP - (brk * (ANGLE_STOP - ANGLE_MAX_REV))
            else:
                drive_state.is_reversing = False
                kit.servo[ESC_CH].angle = ANGLE_STOP
    except Exception as e:
        print(f"Drive WS error: {e}")
        kit.servo[ESC_CH].angle = ANGLE_STOP

@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    global picam2, camera_active
    await websocket.accept()
    print("🎥 Video WS Connected", flush=True)
    
    if picam2 is None:
        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            camera_active = True
            print("✅ カメラを新規に開始しました。")
        except Exception as e:
            print(f"❌ カメラ起動失敗: {e}")
            if "Running" in str(e): camera_active = True
            else: await websocket.close(1011); return
    else:
        print("🔄 既存のカメラを再利用します。")
        camera_active = True

    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)
            if frame is not None:
                success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if success:
                    jpg_as_text = base64.b64encode(buf).decode('utf-8')
                    await websocket.send_json({"type": "video", "data": jpg_as_text})
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"❌ Video Loop End: {e}")

if __name__ == "__main__":
    import uvicorn
    # 最も確実に lifespan を呼ぶ起動方法
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")