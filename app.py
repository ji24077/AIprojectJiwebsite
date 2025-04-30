from flask import Flask, request, render_template, jsonify
import os
from model import SwinModel
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기

# 업로드 폴더가 없으면 생성
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 모델 초기화
model = SwinModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 파일 저장 방식
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 두 가지 방식으로 예측 수행
            # 1. 파일 경로로 예측
            predicted_class1, confidence1 = model.predict(filepath)
            
            # 2. 메모리에서 직접 처리
            image = Image.open(file.stream).convert('RGB')
            predicted_class2, confidence2 = model.predict(image)
            
            # 결과 비교 (디버깅용)
            print(f"파일 경로 방식 결과: {predicted_class1}, {confidence1:.2f}%")
            print(f"메모리 처리 방식 결과: {predicted_class2}, {confidence2:.2f}%")
            
            # 파일 삭제
            os.remove(filepath)
            
            # 결과 반환 (메모리 처리 방식 결과 사용)
            result = {
                'is_ai_generated': bool(predicted_class2),
                'confidence': confidence2,
                'message': 'AI 생성' if predicted_class2 else '실제 이미지'
            }
            return jsonify(result)
            
        except Exception as e:
            print(f"예측 중 오류 발생: {str(e)}")  # 디버깅용 출력
            return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': '요청한 리소스를 찾을 수 없습니다.'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    print("Flask 애플리케이션 시작...")
    print(f"업로드 폴더: {app.config['UPLOAD_FOLDER']}")
    print(f"허용된 파일 확장자: {ALLOWED_EXTENSIONS}")
    app.run(debug=True, port=5002) 