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
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

    def predict(self, image):
        try:
            # 이미지 전처리
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 모든 이미지를 224x224로 리사이즈
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 이미지가 이미 PIL.Image 객체인 경우
            if isinstance(image, Image.Image):
                print("PIL.Image 객체 처리 중...")
                # 이미지가 RGB가 아닌 경우 변환
                if image.mode != 'RGB':
                    print(f"이미지 모드 변환: {image.mode} -> RGB")
                    image = image.convert('RGB')
                with torch.no_grad():
                    image = transform(image).unsqueeze(0)
            
            # 이미지가 파일 경로인 경우
            elif isinstance(image, str):
                print(f"파일 경로 처리 중: {image}")
                with Image.open(image) as img:
                    image = img.convert('RGB')
                    with torch.no_grad():
                        image = transform(image).unsqueeze(0)
            
            # 이미지가 torch.Tensor인 경우
            elif isinstance(image, torch.Tensor):
                print("torch.Tensor 처리 중...")
                with torch.no_grad():
                    if image.dim() == 3:  # [C, H, W]
                        image = image.unsqueeze(0)  # [1, C, H, W]
                    if image.shape[2:] != (224, 224):  # 크기가 224x224가 아닌 경우
                        image = transforms.functional.resize(image, (224, 224))
            
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 모델 예측
            self.model.eval()
            device = next(self.model.parameters()).device
            image = image.to(device)
            
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
            
        except FileNotFoundError as e:
            print(f"파일을 찾을 수 없습니다: {str(e)}")
            raise ValueError(f"파일을 찾을 수 없습니다: {str(e)}")
        except Image.UnidentifiedImageError as e:
            print(f"이미지 파일을 인식할 수 없습니다: {str(e)}")
            raise ValueError(f"이미지 파일을 인식할 수 없습니다: {str(e)}")
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            print(f"오류 타입: {type(e)}")
            print(f"오류 상세: {str(e)}")
            raise

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