from flask import Flask, request, render_template, jsonify
import os
from model import SwinModel
from werkzeug.utils import secure_filename

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
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 예측 수행
            result = model.predict(filepath)
            
            # 파일 삭제 (선택사항)
            os.remove(filepath)
            
            return jsonify({
                'is_ai_generated': result['is_ai_generated'],
                'confidence': result['confidence'],
                'message': 'AI 생성' if result['is_ai_generated'] else '실제 이미지'
            })
            
        except Exception as e:
            return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500
            
    return jsonify({'error': '허용되지 않는 파일 형식입니다'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5002) 