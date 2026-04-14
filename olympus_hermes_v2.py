import asyncio
import socket
import time
import base64
import cv2
from fastapi import FastAPI, WebSocket
from adafruit_servokit import ServoKit
from picamera2 import Picamera2
import board

app = FastAPI()
kit = ServoKit(channels=16)

# ==========================================
# 🔧 物理・制御パラメータ
# ==========================================
PAN_CH, TILT_CH = 1, 0  # パンチルト
ESC_CH, STR_CH  = 2, 3  # 走行・ステア

# サーボパルス幅初期化
for ch in [PAN_CH, TILT_CH, ESC_CH, STR_CH]:
    kit.servo[ch].set_pulse_width_range(500, 2500)

# 走行設定
ANGLE_STOP, ANGLE_MAX_FWD, ANGLE_MAX_REV = 90, 95, 85
ANGLE_CENTER, ANGLE_LEFT, ANGLE_RIGHT = 86, 61, 111

# カメラ設定
try:
    print("📸 カメラの初期化を開始します...")
    time.sleep(1.0)  # OSがカメラを落ち着いて認識するまで2秒待つ
    picam2 = Picamera2()
    
    # 接続確認まで少し待機
    time.sleep(0.5)
    
    config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}) # 負荷軽減のためサイズダウン
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"Saturation": 1.5})
    # --- 💡 ホワイトバランスの設定を追加 ---
    # --- 💡 安全なホワイトバランス設定 ---
    #try:
        # まず、この環境で使えるコントロール一覧を確認（デバッグ用）
        #print(picam2.camera_controls)
        
        # libcamera のコントロール定義を直接使って、
        # 「AwbMode を 2 (Indoor の ID) にせよ」と命令します
        #from libcamera import controls
        
        # libcameraの標準的なAWBモードID:
        # 1: Auto, 2: Incandescent, 3: Tungsten, 4: Fluorescent, 5: Indoor, 6: Daylight, 7: Cloudy
        # 室内であれば 5(Indoor) または 4(Fluorescent) が最適です。
        #picam2.set_controls({
        #    "AwbMode": 0,
        #    "ColourGains": (1.8, 1.0)
        #    })
        
    #    print("🎨 カラーゲインをマニュアル調整しました")
    #except Exception as e:
    #    print(f"⚠️ ゲイン設定失敗: {e}")

    print("📸 カメラ(imx219)の起動に成功しました！")
except Exception as e:
    print(f"❌ カメラ起動失敗: {e}")
    
    # 失敗したときのために None を入れておく
    picam2 = None
###
# ==========================================
# 🧠 高度追従エンジン（Jerk制限）
# ==========================================
class SmoothTracker:
    def __init__(self):
        self.curr = [90.0, 90.0]
        self.target = [90.0, 90.0]
        self.vel = [0.0, 0.0]
        self.speed_gain, self.base_accel, self.boost_accel = 0.48, 0.20, 0.65
        self.damping, self.deadband = 0.78, 0.12

    def update_target(self, yaw, pitch):
        self.target[0] = 90 + yaw
        self.target[1] = 90 + pitch

    def drive_step(self):
        for i in range(2):
            error = self.target[i] - self.curr[i]
            accel_limit = self.boost_accel if abs(error) > 3.0 else self.base_accel
            vel_diff = max(-accel_limit, min(accel_limit, (error * self.speed_gain) - self.vel[i]))
            self.vel[i] = (self.vel[i] + vel_diff) * self.damping
            if abs(error) < self.deadband: self.vel[i] *= 0.4
            self.curr[i] += self.vel[i]
            #print(f"Servo Angle -> PAN: {round(self.curr[0], 1)}, TILT: {round(self.curr[1], 1)}", flush=True)
        kit.servo[PAN_CH].angle = max(0, min(180, round(self.curr[0], 1)))
        kit.servo[TILT_CH].angle = max(0, min(180, round(self.curr[1], 1)))

tracker = SmoothTracker()

# ==========================================
# 📡 ネットワークタスク
# ==========================================

# 1. 顔追従UDP受信 & 100Hz駆動ループ
class VisionProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data, addr):
        vals = data.decode().split(',')
        if len(vals) >= 2:
            tracker.update_target(float(vals[0]), float(vals[1]))

    def error_received(self, exc):
        print(f"⚠️ UDP error: {exc}", flush=True)

async def vision_control_loop():
    
    loop = asyncio.get_event_loop()
    
    # OSに登録するだけ
    transport, protocol = await loop.create_datagram_endpoint(
    VisionProtocol,
    local_addr=("0.0.0.0", 5005)
    )
    
async def servo_drive_loop():
    try
        while True:
            tracker.drive_step()
            await asyncio.sleep(0.01)
    except Exception as e:
        print(f"servo_drive_error: {e}")

class DriveState:
    def __init__(self):
        self.is_reversing = False

drive_state = DriveState()

# 2. 走行制御 WebSocket
@app.websocket("/ws/drive")
async def drive_endpoint(websocket: WebSocket):
    #global is_reversing
    await websocket.accept()
    
    #is_reversing = False
    
    try:
        while True:
            data = await websocket.receive_json()　#JSONデータが来るまで待機。OSが通知するまでCPU不使用
            thr, brk, back, neut, steer = data['throttle'], data['brake'], data['isBackMode'], data['isNeutral'], data['steer']
            # ステアリング
            if neut: kit.servo[STR_CH].angle = ANGLE_CENTER
            else:
                s_ang = ANGLE_CENTER + ((steer - 128) / 128 * (ANGLE_RIGHT - ANGLE_CENTER))
                kit.servo[STR_CH].angle = max(min(s_ang, ANGLE_RIGHT), ANGLE_LEFT)
            # ESC
            if back:
                if not drive_state.is_reversing:
                    kit.servo[ESC_CH].angle = ANGLE_MAX_REV
                    await asyncio.sleep(0.1); kit.servo[ESC_CH].angle = ANGLE_STOP
                    await asyncio.sleep(0.1); kit.servo[ESC_CH].angle = ANGLE_MAX_REV
                    drive_state.is_reversing = True
                else: kit.servo[ESC_CH].angle = ANGLE_MAX_REV
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

# 3. 映像配信 WebSocket
@app.websocket("/ws/video")
async def video_endpoint(websocket: WebSocket):
    await websocket.accept()
    if picam2 is None:
        await websocket.close(1011, "Camera not available")
        return
    try:
        while True:
            frame = await asyncio.to_thread(picam2.capture_array)　#カメラ完了まで待機。スレッドが動く間CPU不使用
            #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #_, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            _,buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            jpg_as_text = base64.b64encode(buf).decode('utf-8')
            await websocket.send_json({"type": "video", "data": jpg_as_text})
            await asyncio.sleep(0.03) # 約30fps
    except Exception as e:
        print(f"Video WS error: {e}")

@app.on_event("startup")
async def startup():
    asyncio.create_task(vision_control_loop())
    asyncio.create_task(servo_drive_loop())

if __name__ == "__main__":
    import uvicorn
    import signal

    # サーバー設定
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Ctrl+C を受け取ったときに確実にループを止める設定
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C を検知しました。強制終了シーケンスを開始します...")
    finally:
        # 1. 全ての実行中のタスクをキャンセル
        for task in asyncio.all_tasks():
            task.cancel()
        
        # 2. カメラを確実に止める
        if 'picam2' in locals() or 'picam2' in globals():
            print("📸 カメラを停止中...")
            picam2.stop()
            picam2.close()
            
        print("✅ プロンプトに戻ります。")
