from model import SwinModel
import os

def test_model():
    try:
        # 모델 초기화
        print("모델 초기화 중...")
        model = SwinModel()
        print("모델 초기화 완료")
        
        # 테스트 이미지 경로
        test_image = "test_image.jpg"  # 테스트용 이미지 파일 경로
        
        if not os.path.exists(test_image):
            print(f"테스트 이미지 파일이 없습니다: {test_image}")
            return
        
        # 예측 실행
        print("\n이미지 분류 중...")
        result = model.predict(test_image)
        
        # 결과 출력
        print("\n=== 예측 결과 ===")
        print(f"AI 생성 이미지 여부: {'예' if result['is_ai_generated'] else '아니오'}")
        print(f"신뢰도: {result['confidence']*100:.2f}%")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        raise

if __name__ == '__main__':
    test_model() 