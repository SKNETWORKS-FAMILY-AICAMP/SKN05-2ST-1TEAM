# SKN05-2ST-1TEAM

## 2차 프로젝트 - 고객 이탈 분석 및 예측
> 기간 2024.10.16 ~ 2024.10.17

## 팀 소개

### 팀명
> "윤관님은 아파요"

### 팀원
- 최영민
- 박찬규
- 배윤관
- 서장호
------------

### 프로젝트 개요
- 소개
   - 현재 국내 스마트폰 사용률은 60대 이상을 제외하면, 모든 연령대에서 98% 이상으로 상당히 높다.
   - 그만큼 이동통신사업에서 고객 이탈을 예측하고 이탈 고객의 행동 및 패턴을 분석하여 이를 방지하는 전략은 매우 중요하다.
   - 우리 팀은 Kaggle의 cell2cell 데이터셋을 활용하여 통신사 이탈 고객들을 분석하여 향후 고객들의 이탈 여부를 예측하는 모델을 구현하고자 한다.

- 주요 목표
   1. 전처리된 데이터를 학습하여 이탈 고객을 예측하는 최적의 모델을 구현한다.
   2. 부가적으로 예측 결과를 바탕으로 고객 이탈에 가장 큰 영향을 미치는 요인을 파악하여 향후 이탈을 방지하도록 한다.

### 전처리
1. 결측치 제거
   - 최대한 KNNImputer를 사용해 결측치를 채우고, 이후에 결측치 제거를 수행
      - KNNImputer를 사용한 경우와 하지 않은 경우의 차이가 크진 않았지만,
2. 파생 변수 생성
   - AgeHH1과 AgeHH2를 통합하여 새로운 피쳐인 Age를 생성
   - ServiceArea가 너무 세분화 되어 있기 때문에 이를 통합한 피쳐인 Division 생성
   - 고유 가입자 수(UniqueSubs)와 활성화된 가입자 수(ActiveSubs)의 비율인 SubsRatio 생성
3. 중요 피쳐 선정
   - 너무 많은 피쳐들이 있어서 학습에 방해가 된다고 생각하여 중요한 피쳐들을 선정하는 작업을 수행하였다.
   - 일단 1차적으로 Chi-Square와 p-value를 이용하여 걸러내었다.
   - 그리고 SHAP value 등을 분석하여 중요하다고 여겨지는 피쳐들로 구성된 Case1, 2, 3 리스트를 만들었다.
   - 이 리스트들에 대해 각각 학습을 시킨 후에 평가 결과가 가장 좋은 Case3 피쳐 리스트를 앞으로
   - 학습을 위해 사용될 최종 피쳐 리스트로 선정하였다.
      
### 모델 분석 및 평가
1. 학습을 수행한 모델
   - Logistic Regression
   - KNN
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost
   - MLP

### 평가 방법
- 이번 프로젝트는 고객의 이탈 여부를 예측하는 것이 최우선 목표이므로 이탈(1)에 대한 f1-score를 중점적으로 모델을 평가하였다.

### 평가 결과
- 오버 또는 언더 샘플링을 따로 수행하지 않은 경우에는 대체로 모델들의 성능이 그리 좋지 않았다.
- f1-score가 가장 높은 것이 0.3 정도였다.
- 아마도 0(이탈 X)에 비해 1(이탈 O)의 데이터가 현저하게 적은 데이터 불균형 때문일 것이다.
- 그래서 우리 팀은 오버 샘플링과 언더 샘플링을 모두 수행해 보았다.
- 오버 샘플링을 한 경우에는 애초에 데이터 양이 충분했는데 거기에 없던 데이터를 생성해내서 그런지 성능이 그다지 좋지 않았다.
- 언더 샘플링을 한 경우에는 거의 대부분의 모델들에서 f1-score가 0.4가 되면서 성능이 개선되었다.
- 그 중에서 가장 성능이 좋은 CatBoost 모델을 최적의 모델로 선정하여 고객 이탈 여부 및 확률을 계산하였다.

### 추가 분석 및 결론
- 이후에 딥러닝 및 10-fold 교차검증을 따로 수행하였다.
- cell2cell 데이터셋 관련 분석 논문에서는 따로 수행되지 않은 최신 모델과 기법들을 활용하였고,
- 그 결과 DNN(Deep Neural Networks) 모델이 f1-score 0.7424로 가장 우수한 성능을 기록했으며,
- RNN, CNN 등의 모델도 강력한 성능을 보였다. 이 결과는 딥러닝 모델이 복잡한 패턴을 학습하는 데 강점을 가지고 있음을 시사한다.

     
### 한 줄 회고
- 최영민 
   > 팀원들이 열정적으로 참여해주어 잘 마무리 할 수 있었습니다. 한가지 평가지표만 고려하는 것이아닌 여러 지표 중 비즈니스 목적에 맞는 평가지표들을 선정하고 이들을 고려하여 발전시키는 것이 중요하다는 것을 느꼈습니다.

- 박찬규
   > 최적의 모델을 구현하기 위해 많은 고려사항을 적용해 봐야한다는것을 느꼈다.

- 배윤관  
   > 뛰어난 팀원들 덕분에 프로젝트를 하는 과정에서 '이렇게도 할 수 있구나'를 느끼면서 많이 배운 것 같습니다.

- 서장호
   > 유능한 팀원들 덕분에 완성할 수 있었다 감사합니다.