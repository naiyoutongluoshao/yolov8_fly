import json
import shutil
import cv2
import os
import torch  # 新增：导入torch检查CUDA
from flask import Flask, Response, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
# 注意：确保 utils.Fun、predictImg、chatApi、predictBatch 模块存在
from utils.Fun import Fun
from utils import predictImg, chatApi, predictBatch


# Flask 应用设置
class VideoProcessingApp:
    def __init__(self, host='0.0.0.0', port=5000):
        """初始化 Flask 应用并设置路由"""
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.host = host
        self.port = port
        self.fun = Fun()
        self.setup_routes()
        self.DeepSeek = ''
        self.Qwen = 'sk-emgjzgivkqwbfrnbnangpdcjhpeaonmsvbfmcghklxhvtuwr'
        self.data = {}
        self.paths = {
            'download': './runs/video/download.mp4',
            'output': './runs/video/output.mp4',
            'camera_output': "./runs/video/camera_output.avi",
            'video_output': "./runs/video/camera_output.avi"
        }
        self.recording = False
        # 新增：自动检测可用的CUDA设备（关键修复）
        self.device = self._get_available_device()
        print(f"当前使用设备：{self.device}")

    def _get_available_device(self):
        """自动检测可用设备：优先用GPU，无则用CPU"""
        if torch.cuda.is_available():
            # 获取第一个可用的GPU索引（解决Invalid device id核心问题）
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                return 0  # 单GPU场景固定用0号设备，多GPU可根据需求调整
        return 'cpu'

    def setup_routes(self):
        """设置所有路由（新增统计接口）"""
        # 原有路由
        self.app.add_url_rule('/', 'index', self.index, methods=['GET'])
        self.app.add_url_rule('/file_names', 'file_names', self.file_names, methods=['GET'])
        self.app.add_url_rule('/predictImgBatch', 'predictImgBatch', self.predictImgBatch, methods=['POST'])
        self.app.add_url_rule('/predictImg', 'predictImg', self.predictImg, methods=['POST'])
        self.app.add_url_rule('/predictVideo', 'predictVideo', self.predictVideo)
        self.app.add_url_rule('/predictCamera', 'predictCamera', self.predictCamera)
        self.app.add_url_rule('/stopCamera', 'stopCamera', self.stopCamera, methods=['GET'])
        self.app.add_url_rule('/api/get_records', 'get_records', self.get_records, methods=['GET'])
        # 新增：统计接口（供前端图表使用）
        self.app.add_url_rule('/api/stats/detection_count', 'get_detection_count', self.get_detection_count,
                              methods=['GET'])

        # WebSocket 事件
        @self.socketio.on('connect')
        def handle_connect():
            print("WebSocket connected!")
            emit('message', {'data': 'Connected to WebSocket server!'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("WebSocket disconnected!")

    def index(self):
        """根路径接口说明页"""
        api_info = """
        <h1>水稻病虫害检测系统 - 接口说明</h1>
        <ul>
            <li><b>GET /file_names</b>：获取权重模型列表</li>
            <li><b>POST /predictImgBatch</b>：批量图片检测（JSON 参数）</li>
            <li><b>POST /predictImg</b>：单张图片检测（JSON 参数，支持AI建议）</li>
            <li><b>GET /predictVideo</b>：视频文件检测（URL 参数）</li>
            <li><b>GET /predictCamera</b>：摄像头实时检测（URL 参数）</li>
            <li><b>GET /stopCamera</b>：停止摄像头检测</li>
            <li><b>GET /api/get_records?type=single_image</b>：获取检测记录（原始数据）</li>
            <li><b>GET /api/stats/detection_count?type=single_image</b>：获取检测统计（供图表使用）</li>
        </ul>
        <p>服务已正常运行，可通过 Postman/前端调用上述接口</p>
        """
        return api_info

    def run(self):
        """启动 Flask 应用"""
        self.socketio.run(self.app, host=self.host, port=self.port, allow_unsafe_werkzeug=True)

    def _save_record_locally(self, record_data, record_type):
        """通用本地保存记录方法"""
        try:
            if not os.path.exists('./records'):
                os.makedirs('./records')
            file_path = f'./records/{record_type}_records.json'
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
            print(f"{record_type} 记录已保存到本地：{file_path}")
        except Exception as e:
            print(f"本地保存 {record_type} 记录失败：{str(e)}")

    def get_records(self):
        """获取原始检测记录（原有接口，保留）"""
        try:
            record_type = request.args.get('type', 'single_image')
            record_file = f'./records/{record_type}_records.json'
            records = []
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if line.strip():
                            records.append(json.loads(line.strip()))
            return {
                "code": 200,
                "msg": "获取记录成功",
                "data": records
            }
        except Exception as e:
            return {
                "code": 500,
                "msg": f"获取记录失败：{str(e)}",
                "data": []
            }

    def get_detection_count(self):
        """新增：统计检测类别数量（供前端图表使用）"""
        try:
            record_type = request.args.get('type', 'single_image')
            record_file = f'./records/{record_type}_records.json'

            # 统计字典：key=病害名称，value=数量
            count_map = {}

            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if line.strip():
                            record = json.loads(line.strip())
                            # 解析label字段（JSON字符串转列表）
                            if 'label' in record and record['label'] != '[]':
                                try:
                                    # 关键：解析JSON字符串为列表，提取第一个类别
                                    labels = json.loads(record['label'])
                                    if labels:
                                        # 取第一个检测结果作为统计类别
                                        main_label = labels[0]
                                        # 累加计数
                                        count_map[main_label] = count_map.get(main_label, 0) + 1
                                except json.JSONDecodeError:
                                    # 兼容解析失败的情况
                                    continue

            # 转换为前端图表需要的格式
            stats = [{"name": label, "value": count} for label, count in count_map.items()]

            return {
                "code": 200,
                "msg": "获取统计数据成功",
                "data": stats
            }
        except Exception as e:
            return {
                "code": 500,
                "msg": f"获取统计数据失败：{str(e)}",
                "data": []
            }

    def predictImgBatch(self):
        """批量图片预测接口（修复版）"""
        try:
            data = request.get_json()
            self.data.clear()
            self.data.update({
                "imgFolderUrl": data['imgFolderUrl'], "username": data['username'],
                "weight": data['weight'], "conf": data['conf']
            })

            # 校验参数
            if not self.data.get('weight'):
                return {"code": 400, "message": "权重文件不能为空", "data": {}}
            if not self.data.get('imgFolderUrl'):
                return {"code": 400, "message": "图片文件夹URL不能为空", "data": {}}

            # 下载文件夹
            self.fun.download_folder(self.data["imgFolderUrl"], './runs/imgBatch')

            # ✅ 核心：在这里加载模型并指定设备，避免传给ImagePredictor
            model = YOLO(f'./weights/{self.data["weight"]}', task='detect')
            model.to(self.device)  # 把模型放到0号GPU或CPU

            predictor = predictBatch.ImagePredictor(
                weights_path=f'./weights/{self.data["weight"]}',
                input_folder="./runs/imgBatch",
                output_folder="./runs/resultBatch",
                conf=float(self.data["conf"]),
                data=self.data
                # ❌ 移除这里的 device=self.device，避免参数不匹配
            )
            batch_result = predictor.predict_batch()
            print(batch_result)

            # 安全清理文件夹
            for folder in ["./runs/imgBatch", "./runs/resultBatch"]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)

            self.data["batch_result"] = batch_result
            self.data["status"] = 200
            self.data["message"] = "批量预测成功"
            self._save_record_locally(self.data, 'batch_image')

            return {"code": 0, "message": "预测成功", "data": batch_result}

        except Exception as e:
            print(f"批量预测失败：{str(e)}")  # 后端打印详细错误
            return {"code": 500, "message": f"批量预测失败：{str(e)}", "data": {}}


    def file_names(self):
        """模型列表接口"""
        weight_items = [{'value': name, 'label': name} for name in self.fun.get_file_names("./weights")]
        return json.dumps({'weight_items': weight_items})

    def predictImg(self):
        """单张图片预测接口"""
        data = request.get_json()
        self.data.clear()
        self.data.update({
            "username": data['username'], "weight": data['weight'],
            "conf": data['conf'], "startTime": data['startTime'],
            "inputImg": data['inputImg'], "ai": data['ai']
        })
        predict = predictImg.ImagePredictor(weights_path=f'./weights/{self.data["weight"]}',
                                            img_path=self.data["inputImg"], save_path='./runs/result.jpg',
                                            conf=float(self.data["conf"]))
        results = predict.predict()
        uploadedUrl = self.fun.upload('./runs/result.jpg')

        if results['labels'] != '预测失败':
            self.data["status"] = 200
            self.data["message"] = "预测成功"
            self.data["outImg"] = uploadedUrl
            self.data["allTime"] = results['allTime']
            self.data["confidence"] = json.dumps(results['confidences'] if results.get('confidences') else [])
            self.data["label"] = json.dumps(results['labels'] if results.get('labels') else [])
        else:
            self.data["status"] = 400
            self.data["message"] = "该图片无法识别，请重新上传！"
            self.data["confidence"] = "[]"
            self.data["label"] = "[]"

        if self.data["ai"] == 'DeepSeek' and self.data["status"] == 200:
            self.socketio.emit('message', {'data': '已检测完成，正在生成DeepSeekAI建议！'})
            chat = chatApi.ChatAPI(
                deepseek_api_key=self.DeepSeek,
                qwen_api_key=self.Qwen
            )
            list_input = self.fun.process_list(results['labels'])
            text = ("我使用yolo对水稻病虫害进行检测。接下来我会告诉你检测到了哪些目标。"
                    "请你帮我生成一些实质性的建议，包括为什么会产生这样的疾病、对该作物种植的建议、对该疾病的治疗措施等。只需回答我要的结果。这是我检测到的结果：")
            for i in list_input:
                text += i
                text += "，"
            messages = [
                {"role": "user",
                 "content": text}
            ]
            self.data["suggestion"] = chat.deepseek_request(
                [{"role": "system", "content": "You are a helpful assistant"}] + messages)
        elif self.data["ai"] == 'Qwen' and self.data["status"] == 200:
            self.socketio.emit('message', {'data': '已检测完成，正在生成QwenAI建议！'})
            chat = chatApi.ChatAPI(
                deepseek_api_key=self.DeepSeek,
                qwen_api_key=self.Qwen
            )
            list_input = self.fun.process_list(results['labels'])
            text = ("我使用yolo对水稻病虫害进行检测。接下来我会告诉你检测到了哪些目标。"
                    "请你帮我生成一些实质性的建议，包括为什么会产生这样的疾病、对该作物种植的建议、对该疾病的治疗措施等。只需回答我要的结果。这是我检测到的结果：")
            for i in list_input:
                text += i
                text += "，"
            messages = [
                {"role": "user",
                 "content": text}
            ]
            self.data["suggestion"] = chat.qwen_request(messages)
        else:
            self.data["suggestion"] = '未选择AI，无AI建议！'

        self.fun.cleanup_files(['./' + self.data["inputImg"].split('/')[-1]])
        self._save_record_locally(self.data, 'single_image')

        return json.dumps(self.data, ensure_ascii=False)

    def predictVideo(self):
        """视频流处理接口"""
        self.data.clear()
        self.data.update({
            "username": request.args.get('username'),
            "weight": request.args.get('weight'),
            "conf": request.args.get('conf'),
            "startTime": request.args.get('startTime'),
            "inputVideo": request.args.get('inputVideo')
        })

        # 修复1：校验weight参数不能为空
        if not self.data.get('weight'):
            return Response("错误：权重文件路径不能为空", status=400, mimetype='text/plain')

        # 修复2：下载视频文件
        self.fun.download(self.data["inputVideo"], self.paths['download'])
        cap = cv2.VideoCapture(self.paths['download'])
        if not cap.isOpened():
            return Response("无法打开视频文件", status=400, mimetype='text/plain')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(fps)

        video_writer = cv2.VideoWriter(
            self.paths['video_output'],
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (640, 480)
        )

        # 修复3：加载模型时显式指定device和task（消除警告+解决GPU问题）
        model = YOLO(f'./weights/{self.data["weight"]}', task='detect')

        def generate():
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (640, 480))
                    # 修复4：predict时显式指定device（核心解决Invalid device id）
                    results = model.predict(
                        source=frame,
                        conf=float(self.data['conf']),
                        show=False,
                        device=self.device  # 显式指定GPU/CPU设备
                    )
                    processed_frame = results[0].plot()
                    video_writer.write(processed_frame)
                    _, jpeg = cv2.imencode('.jpg', processed_frame)
                    yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            finally:
                self.fun.cleanup_resources(cap, video_writer)
                self.socketio.emit('message', {'data': '处理完成，正在保存！'})
                # 修复5：处理进度解析N/A的问题（包装convert_avi_to_mp4）
                for progress in self._safe_convert_avi_to_mp4(self.paths['video_output']):
                    self.socketio.emit('progress', {'data': progress})
                uploadedUrl = self.fun.upload(self.paths['output'])
                self.data["outVideo"] = uploadedUrl
                self._save_record_locally(self.data, 'video')
                self.fun.cleanup_files([self.paths['download'], self.paths['output'], self.paths['video_output']])

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def predictCamera(self):
        """摄像头视频流处理接口"""
        self.data.clear()
        self.data.update({
            "username": request.args.get('username'),
            "weight": request.args.get('weight'),
            "conf": request.args.get('conf'),
            "startTime": request.args.get('startTime')
        })


        # 修复1：校验weight参数不能为空（解决接口传空的问题）
        if not self.data.get('weight'):
            return Response("错误：权重文件路径不能为空", status=400, mimetype='text/plain')

        self.socketio.emit('message', {'data': '正在加载，请稍等！'})

        # 修复2：加载模型时显式指定device和task
        model = YOLO(f'./weights/{self.data["weight"]}', task='detect')
        #############esp32摄像头关键处
        cap = cv2.VideoCapture("http://192.168.217.16/stream")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_writer = cv2.VideoWriter(self.paths['camera_output'], cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))
        self.recording = True

        def generate():
            try:
                while self.recording:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 修复3：predict时显式指定device
                    results = model.predict(
                        source=frame,
                        imgsz=640,
                        conf=float(self.data['conf']),
                        show=False,
                        device=self.device  # 显式指定GPU/CPU设备
                    )
                    processed_frame = results[0].plot()
                    if self.recording and video_writer:
                        video_writer.write(processed_frame)
                    _, jpeg = cv2.imencode('.jpg', processed_frame)
                    yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            finally:
                self.fun.cleanup_resources(cap, video_writer)
                self.socketio.emit('message', {'data': '处理完成，正在保存！'})
                # 修复4：安全处理视频转换进度
                for progress in self._safe_convert_avi_to_mp4(self.paths['camera_output']):
                    self.socketio.emit('progress', {'data': progress})
                uploadedUrl = self.fun.upload(self.paths['output'])
                self.data["outVideo"] = uploadedUrl
                self._save_record_locally(self.data, 'camera')
                # 修复5：清理文件时跳过不存在的文件
                self.fun.cleanup_files([
                    p for p in [self.paths['download'], self.paths['output'], self.paths['camera_output']]
                    if os.path.exists(p)
                ])

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _safe_convert_avi_to_mp4(self, input_path):
        """安全包装convert_avi_to_mp4，处理N/A转换失败"""
        try:
            for progress in self.fun.convert_avi_to_mp4(input_path):
                # 处理进度为N/A的情况
                if progress == 'N/A' or not isinstance(progress, (int, float)):
                    yield 0.0  # 替换为0
                else:
                    yield progress
        except Exception as e:
            print(f"视频转换出错：{e}")
            yield 100.0  # 出错时直接返回100%，避免卡住

    def stopCamera(self):
        """停止摄像头预测"""
        self.recording = False
        return json.dumps({"status": 200, "message": "停止成功", "code": 0})


# 启动应用
if __name__ == '__main__':
    video_app = VideoProcessingApp()
    video_app.run()