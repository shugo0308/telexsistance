import cv2
import mediapipe as mp
import subprocess
import numpy as np
import asyncio
import base64
import socket
import uvicorn
import json
#import math
import board
import neopixel
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse,FileResponse
from contextlib import asynccontextmanager

# --- ネットワーク・デバイス設定 ---
DEST_IP = "192.168.0.41" #ラズパイ5のIPアドレス
DEST_PORT = 5005         #ラズパイ5のポート番号
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# LEDリング設定 (GPIO 18 / 16灯)
NUM_PIXELS = 16
PIXEL_PIN = board.D18
pixels = neopixel.NeoPixel(PIXEL_PIN, NUM_PIXELS, brightness=0.3, auto_write=False)


# ==========================================
# 通信管理用クラス
# ==========================================
class ConnectionManager:
    def __init__(self): self.connections = []                                     #通信を開始したWebsocket接続を格納する為のリストの作成
    async def connect(self, ws): await ws.accept(); self.connections.append(ws)   #リストにWebsocket接続を登録
    def disconnect(self, ws): self.connections.remove(ws)                         #Websocket接続が切れた場合にリストから削除
    async def broadcast_json(self, data):
        #for c in self.connections:
        #    try: await c.send_json(data)
        #    except: pass
        stale = [] # 通信が切れている「幽霊会員」を入れるリスト
        
        for c in list(self.connections):
            # データを送信。0.5秒以内に送れなければ「遅延」とみなしてタイムアウトさせる
            try:
                await asyncio.wait_for(c.send_json(data), timeout=0.5)
            except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
                stale.append(c) # エラーが出た接続(死んだ接続)をメモする
        # 通信が死んでいる接続を名簿から掃除する
        for c in stale:
            if c in self.connections:
                self.connections.remove(c)

manager = ConnectionManager()

# ==========================================
# 顔の画像から角度と距離を変換するクラス
# ==========================================
class PoseEstimator:
    #基準データと3Dモデルの定義
    def __init__(self, width=640, height=480):
        self.w, self.h = width, height
        self.last_raw_y, self.filtered_yaw, self.filtered_pitch = 0.0, 0.0, 0.0
        
        # 基準（オフセット）が設定されたかどうかを管理
        self.is_calibrated = False 
        self.offset_yaw, self.offset_pitch = 0.0, 0.0
        
        # 感度の初期設定
        #self.calib_right = 25.0
        #self.calib_left = -25.0 
        
        self.deadzone, self.max_velocity, self.base_alpha = 0.2, 4.0, 0.02
        self.FOCAL_LENGTH_PIXELS = 840.0 
        
        # PnP用モデル点
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻先
            (0.0, -33.0, -30.0),         # 顎 (鼻より30mm奥にある)
            (-22.5, 17.0, -30.0),        # 左目端
            (22.5, 17.0, -30.0),         # 右目端
            (-15.0, -15.0, -30.0),       # 口角左
            (15.0, -15.0, -30.0)         # 口角右
        ], dtype="double")
        
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([[self.FOCAL_LENGTH_PIXELS, 0, center[0]], 
                                       [0, self.FOCAL_LENGTH_PIXELS, center[1]], 
                                       [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4, 1))
    #ゼロ点合わせ
    def set_center(self, ry, rp):
        # 現在の「生の角度」を基準点として保存
        self.offset_yaw = ry
        self.offset_pitch = rp
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0
        self.last_raw_y = 0.0
        self.is_calibrated = True # ここで初めて計算を有効にする
        print(f"Center Set: Offset Y={ry:.1f}")
    #SolvePnPの計算
    def get_angles_and_distance(self, face_landmarks, manual_offset):
        # 1. 角度計算 (PnP)
        image_points = np.array([
            (face_landmarks.landmark[1].x * self.w, face_landmarks.landmark[1].y * self.h),
            (face_landmarks.landmark[152].x * self.w, face_landmarks.landmark[152].y * self.h),
            (face_landmarks.landmark[33].x * self.w, face_landmarks.landmark[33].y * self.h),
            (face_landmarks.landmark[263].x * self.w, face_landmarks.landmark[263].y * self.h),
            (face_landmarks.landmark[61].x * self.w, face_landmarks.landmark[61].y * self.h),
            (face_landmarks.landmark[291].x * self.w, face_landmarks.landmark[291].y * self.h)
        ], dtype="double")
        #PnPアルゴリズム。MediaPipeが検出した顔のパーツの2D位置と自分が持っている3Dモデルの照らし合わせ
        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        
        # 距離計算は角度に関係なく行う
        p1 = np.array([face_landmarks.landmark[234].x * self.w, face_landmarks.landmark[234].y * self.h])
        p2 = np.array([face_landmarks.landmark[454].x * self.w, face_landmarks.landmark[454].y * self.h])
        pixel_dist = np.linalg.norm(p1 - p2)
        dist_mm = (145.0 * self.FOCAL_LENGTH_PIXELS) / pixel_dist if pixel_dist > 0 else 0 #距離の推定(目尻の距離)

        if not success: return 0.0, 0.0, 0.0, 0.0, dist_mm

        # 生の角度を算出
        rmat, _ = cv2.Rodrigues(rvec)
        r_yaw = np.degrees(np.arctan2(rmat[0, 2], rmat[2, 2]))
        r_yaw = (r_yaw + 180 + 180) % 360 - 180
        r_pitch = np.degrees(np.arctan2(rmat[1, 2], np.sqrt(rmat[1, 0]**2 + rmat[1, 1]**2)))
        
        #診断用ログ
        #print(f"RAW: {r_yaw:.1f}")

        # まだキャリブレーションされていないなら角度 0 を返す
        if not self.is_calibrated:
            return 0.0, 0.0, r_yaw, r_pitch, dist_mm

        # 2. オフセット適用とフィルタリング(ローパスフィルタ)
        raw_y = ((r_yaw - self.offset_yaw + manual_offset + 180) % 360 - 180)
        
        diff_y = raw_y - self.last_raw_y
        if abs(diff_y) > self.max_velocity: raw_y = self.last_raw_y + np.sign(diff_y) * self.max_velocity
        
        #alpha = max(self.base_alpha * (1.0 - abs(raw_y)/100.0), 0.2)
        # ✅ 修正案：誤差が大きいときはalphaを大きく（速く追従）
        alpha = min(self.base_alpha + abs(raw_y) / 100.0 * 0.3, 0.5)
        self.filtered_yaw = alpha * raw_y + (1.0 - alpha) * self.filtered_yaw
        self.last_raw_y = raw_y
        
        # --- [修正点] 左右対称・固定感度ロジック ---
        # calib_left / right を使わず、一律で「25度首を振ったら最大(38度)」に固定します
        # これにより、ボタン操作ミスによる感度の左右差が物理的に発生しなくなります
        
        fixed_denom = 40.0  # この値を大きくすると動きがマイルドに、小さくするとクイックになります
        target_max = 38.0
        
        # 単純な割り算でマッピング（左右の区別をなくす）
        y_out = (self.filtered_yaw / fixed_denom) * target_max
        
        # 最終リミッター
        y_out = max(-38.0, min(38.0, y_out))
        
        # get_angles_and_distance 内の Pitch 計算部分
        # 生の pitch (r_pitch) から、中心設定時の offset_pitch を引く
        p_out = r_pitch - self.offset_pitch
        # 戻り値も補正後の p_out を使うように変更
        return y_out, p_out, r_yaw, r_pitch, dist_mm

# ==========================================
# 🧠 AI タスク本体
# ==========================================
async def ai_task():
    mp_face_mesh = mp.solutions.face_mesh
    estimator = PoseEstimator()
    
    #HERMES_IP = "192.168.0.41" # エルメスの実際のIP
    #カメラの起動(rpicam-vidの実行)
    cmd = [
        "rpicam-vid", "-t", "0",             #タイムアウトは無し
        "--width", "640", "--height", "480", #解像度640x480 
        "--inline", "--codec", "mjpeg",      #MJPEG形式で動画を取り出し
        "--framerate", "30",                 #フレームレートは30fpsに設定
        "-o", "-"
        ]
    #pythonが読み取れるパイプに繋ぎ替える。rpicam-vidが出すメッセージはゴミ箱行き(subprocessで非同期処理)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        #MediaPipeにて顔をランドマーク(目、鼻、口等の点)を検出するためのエンジンの初期化
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,                #検出最大人数:１人
            refine_landmarks=True,          #瞳や唇の細かな動きを精密に計算するモードの設定(TrueでモードON)
            min_detection_confidence=0.6,   #顔であると判断する為の合格ライン(60%に設定)
            min_tracking_confidence=0.6     #検出した顔を追従し続ける為の合格ライン(60%に設定)
        ) as face_mesh:
            buffer = b"" #カメラから届くデータのバッファを空にする
            while True:
                #カメラからのデータを最大4096バイト分読み込みし、chunkに格納。スレッドで実行してブロッキング処理を防止
                chunk = await asyncio.to_thread(proc.stdout.read, 65536)
                if not chunk:
                    break
                #カメラデータをbufferに継ぎ足し
                buffer += chunk
                #a, b = buffer.find(b'\xff\xd8'), buffer.find(b'\xff\xd9')
                #if a == -1 and b == -1:
                #    continue
            
                #jpg, buffer = buffer[a:b+2], buffer[b+2:]
                #JPEGデータの開始位置を探し、位置をaに格納
                a = buffer.find(b"\xff\xd8")
                #開始位置が無ければバッファを空にして、次のループへ戻る
                if a == -1:
                    buffer = b""
                    continue
                #JPEGデータの終点データの位置をaの2バイトの位置から検索し、終了位置をbに格納
                b = buffer.find(b"\xff\xd9", a + 2)
                #終点位置が無ければ開始マークより前のバッファを空にして、次のループへ戻って続きのデータを待つ
                if b == -1:
                    buffer = buffer[a:]
                    continue
                #JPEGデータの切り出し、残りの部分はbufferに保持
                jpg, buffer = buffer[a:b + 2], buffer[b + 2:]
                #jpgデータをNumpyライブラリで数値の配列に並び替えを行い、BGRのピクセルデータに変換(スレッドで実行) 
                image = await asyncio.to_thread(
                cv2.imdecode, np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                #ピクセルデータが無ければ次のフレームの処理に移る
                if image is None:
                    continue
                        
                # 顔の検出(MediaPipe)
                rgb   = await asyncio.to_thread(cv2.cvtColor, image, cv2.COLOR_BGR2RGB) #ピクセルデータをBGRからRGBに変換
                results = await asyncio.to_thread(face_mesh.process, rgb)               #変換したピクセルデータをMediapipeの解析に渡す
                #最新の解析データを各変数に格納
                y_out = app.state.last_y_out
                r_pitch = app.state.last_pitch_out
                out_dist = app.state.current_distance
                
                #顔が検出されたときの処理
                if results.multi_face_landmarks:
                    t_start = time.perf_counter() #T1計測開始
                    
                    #print("Face Detected!") # 顔検出デバッグ用
                    
                    face = results.multi_face_landmarks[0]
                    #角度と距離の計算(PoseEstimator呼び出し)
                    y, p, r_y, r_p, dist = estimator.get_angles_and_distance(face, app.state.manual_offset_y)
                    
                    current_req = app.state.mode_request
                    if current_req:
                        app.state.mode_request = None
                        if current_req == "center":
                            estimator.set_center(r_y, r_p)
                        print(f"Executing Mode: {current_req}")
                        
                    # 生の計算値に左右反転を適用
                    y_out = -y 
                    
                    r_pitch, out_dist = p, dist
                    app.state.current_distance = out_dist
                    app.state.last_pitch_out = r_pitch
                    
                    # --- [追加] 定量的評価用のログ出力 ---
                    # 形式: タイムスタンプ, 計算角度(y), 送信角度(y_out), 推定距離
                    t_now = time.perf_counter()
                    t1_latency = (t_now - t_start) * 1000 # ミリ秒換算
                    #print(f"[DATA_LOG] {time.time():.3f}, raw_y:{y:.2f}, cmd_y:{y_out:.2f}, dist:{dist:.1f}, T1:{t1_latency:.2f}ms")
                    
                    
                    # LED自動調光: 距離300mm(0.1暗) 〜 1000mm(0.8明)
                    bright = np.clip((dist - 300) / 700 * 0.7 + 0.1, 0.1, 0.8)
                    pixels.brightness = bright
                    pixels.fill((255, 245, 210)) 
                    pixels.show()
                    
                    #エルメスへ顔の角度(左右(Yow)上下(Roll)と送信時刻を送信
                    #print(f"Sending UDP: {y_out}, {r_pitch}") # 顔検出デバッグ用
                    t1_athena = time.time() #認識完了・送信直前の時刻
                    message = f"{y_out},{r_pitch},{t1_athena}"
                    try:
                        udp_sock.sendto(message.encode(), (DEST_IP, DEST_PORT))
                    except Exception as e:
                        print(f"UDP Send Error: {e}")
            
                # ブラウザ配信用にjpgデータをエンコード(圧縮)する
                _, buf = await asyncio.to_thread(
                    cv2.imencode, '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50]
                    )
                #ブラウザへの配信
                await manager.broadcast_json({
                    "img": base64.b64encode(buf).decode('utf-8'), 
                    "y": y_out,
                    "p": r_pitch,
                    "dist": out_dist,
                    "server_ts": time.time() * 1000 # T3計測用：サーバー送信時刻(ms)
                    })
                await asyncio.sleep(0.01) 			#10msec処理を止めて、他の処理を片づける
    finally:
        #強制終了やエラーでパイプが残っている場合
        if proc.stdout is not None:
            proc.stdout.close()	#カメラからデータを受け取っていたパイプを切断
        #カメラプログラムが動いている場合
        if proc.poll() is None:
            proc.terminate()	#カメラプログラムに終了処理をする
            try:
                proc.wait(timeout=2)	#2秒間待つ
            except subprocess.TimeoutExpired:
                proc.kill()				#反応無ければ強制終了

# ==========================================
# 🔁 AIタスク　自動再起動ラッパー
# ==========================================
async def ai_task_runner():
    while True:
        try:
            print("▶ ai_task を起動します")
            await ai_task()
            # ai_task が正常終了（chunk なし）したら再起動
            print("⚠️ ai_task が終了しました。再起動します...")
        except Exception as e:
            print(f"❌ ai_task がクラッシュしました: {e}")
            print("🔄 1秒後に再起動します...")
        await asyncio.sleep(1.0)

# --- Lifespan (起動・終了処理) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 起動時の処理
    print("🚀 Athena System Starting...")
    
    app.state.mode_request = None        						#ブラウザからのコマンドの一時保存用
    app.state.manual_offset_y = 0.0      						#ユーザーが手動で調整した角度補正値用
    app.state.current_distance = 0.0     						#最後に計算したユーザーの顔までの距離
    app.state.last_y_out = 0.0           						#エルメスに送信した左右の角度
    app.state.last_pitch_out = 0.0       						#エルメスに送信した上下の角度
    app.state.stable_y = 0.0             						#不感帯を設定した後の安定した左右角度
    app.state.is_moving = False          						#ユーザーが回転中か静止中か判別するフラグ
    app.state.ai_task = asyncio.create_task(ai_task_runner())	# AIタスクをバックグランドへ投げる
    
    yield
    # システムシャットダウン時の処理
    print("🛑 Athena System Shutting down...", flush=True)
    
    app.state.ai_task.cancel() 		#AIタスクの中止命令
    try:
        await app.state.ai_task 	#AIタスクの停止を待つ
    except asyncio.CancelledError:	#キャンセル報告を受け止めてプログラムを正常終了させる
        print("✅ Background task cancelled.")
            
    # LEDを消灯して終了
    pixels.fill((0, 0, 0))
    pixels.show()
    print("👋 Goodbye.")

# アプリの定義 (lifespanを渡す)
app = FastAPI(lifespan=lifespan)

#index_olympus.htmlファイルの実行
@app.get("/")
async def get(): return FileResponse("index_olympus.html")

# 3. モード切替（補正コマンド）受け取り口の確認
@app.get("/mode")
async def set_mode(request: str):
    # index_athena.html の fetch(`/mode?request=${mode}`) に対応
    app.state.mode_request = request
    print(f"DEBUG: Control Request Received: {request}")
    return {"status": "ok", "mode": request}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws) #ConnectionManagerに接続を登録
    try:
        while True:
            data = await ws.receive_text() #ブラウザからのメッセージを待つ
            msg = json.loads(data) #届いたメッセージを変換
            #if msg["type"] == "mode": app.state.mode_request = msg["value"]
            #if msg["type"] == "offset": app.state.manual_offset_y = msg["value"]
            msg_type = msg.get("type")
            #届いたメッセージの内容を見て処理を振り分け
            if msg_type == "mode":
                app.state.mode_request = msg.get("value") #モード切替命令を登録　
            elif msg_type == "offset":
                value = msg.get("value")
                if isinstance(value, (int, float)):
                    app.state.manual_offset_y = value #左右の微調整値(オフセット)を反映
    #切断の処理(接続リストの削除)                
    except WebSocketDisconnect: manager.disconnect(ws)

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

