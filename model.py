import torch
import torch.nn as nn
from transformers import SwinConfig, SwinForImageClassification, AutoImageProcessor
from PIL import Image
import torchvision.transforms as transforms

class SwinModel:
    def __init__(self):
        # 체크포인트와 일치하는 모델 구성
        config = SwinConfig(
            image_size=224,
            patch_size=4,
            num_channels=3,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.0,
            num_labels=2
        )
        
        # 모델 초기화
        self.model = SwinForImageClassification(config)
        
        # CPU에서 사전 훈련된 가중치 로드
        state_dict = torch.load("Swinmodel.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 이미지 전처리기 초기화
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def predict(self, image):
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        
        # 모델 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=image)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
            
            # 디버깅 출력 추가
            print(f"모델 출력 로짓: {logits}")
            print(f"소프트맥스 확률: {probabilities}")
            print(f"예측된 클래스: {predicted_class}")
            print(f"신뢰도: {confidence:.2f}%")
        
        return predicted_class, confidence

def load_model(model_path):
    try:
        # 모델 인스턴스 생성
        model = SwinModel()
        
        # 모델 상태 사전 로드
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # 평가 모드로 설정
        model.eval()
        
        print("모델이 성공적으로 로드되었습니다.")
        return model
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {str(e)}")
        raise

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(model, image):
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return prediction, confidence 