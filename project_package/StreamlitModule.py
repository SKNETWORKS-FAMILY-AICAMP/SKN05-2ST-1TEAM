import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle

ITEMS_PER_PAGE = 10

CHI2_TABLE_PATH = './database/feature_importance/chi_p_value/chi2_stats.csv'
NUMERIC_MEAN_SHAP_VALUE_PATH = './database/feature_importance/shap/mean_shap_value_numeric.csv'
CATEGORICAL_MEAN_SHAP_VALUE_PATH = './database/feature_importance/shap/mean_shap_value_categorical.csv'
CATEGORICAL_SHAP_SUMMARY_PATH = './database/feature_importance/shap/shap_summary.png'
CATEGORICAL_SHAP_SUMMARY_BAR_PATH = './database/feature_importance/shap/shap_summary_bar.png'
NUMERIC_SHAP_SUMMARY_PATH = './database/feature_importance/shap/shap_summary_numeric.png'
NUMERIC_SHAP_SUMMARY_BAR_PATH = './database/feature_importance/shap/shap_summary_bar_numeric.png'
CUSTOMER_DATA_PATH = './database/churn_results.csv'
PREPROCESSED_DATA_PATH = './database/data.csv'
TRAINED_MODEL_PATH = './model/ranfo_3.pkl'

CATEGORICAL_FEATURES = ['ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner',
                      'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 'OwnsComputer',
                      'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 'MadeCallToRetentionTeam', 
                      "PrizmCode", "Occupation", "MaritalStatus", "Division", 'CreditRating', 'HandsetPrice', 'Homeownership']
FEATURE_LIST = ['Churn', 'MonthlyRevenue', 'TotalRecurringCharge', 'DirectorAssistedCalls',
                'OverageMinutes', 'RoamingCalls', 'PercChangeRevenues',
                'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls',
                'OutboundCalls', 'InboundCalls', 'PeakCallsInOut',
                'CallForwardingCalls', 'CallWaitingCalls', 'MonthsInService',
                'UniqueSubs', 'SubsRatio', 'Handsets', 'CurrentEquipmentDays',
                'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable',
                'BuysViaMailOrder', 'RespondsToMailOffers', 'HasCreditCard',
                'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode_Other',
                'PrizmCode_Rural', 'PrizmCode_Suburban', 'PrizmCode_Town',
                'MaritalStatus_No', 'MaritalStatus_Unknown', 'MaritalStatus_Yes',
                'Division_midwest', 'Division_northeast', 'Division_south',
                'Division_west', 'Age']









####################### 데이터 fetch 함수 #######################
def fetch_customer_table(file_path=CUSTOMER_DATA_PATH):
    # 고위험 순으로 정렬된 데이터 로드
    customer_table = pd.read_csv(CUSTOMER_DATA_PATH)

    return customer_table

@st.cache_resource
def fetch():
    # 모델 로드
    with open(TRAINED_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    
    
    # 고위험군 고객 테이블
    customer_table = fetch_customer_table()


    fetch_result = {
        'customer_table' : customer_table,
        'model' : model
    }

    # return customer_table, model
    return fetch_result















####################### 화면 구성 함수 #######################

################### 이탈 고위험군 고객 테이블 구성 ###################
# 고객 테이블 1 페이지 출력
def show_customer_pages(customer_table):
    # 총 페이지 수 계산
    total_pages = (len(customer_table) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        
    # 페이지 설정
    if 'page' not in st.session_state:
        st.session_state.page = 1
    page = st.session_state.page
    if page < 1:
        st.session_state.page = 1
        page = 1
    elif page > total_pages:
        st.session_state.page = total_pages
        page = total_pages

    # 현재 페이지의 아이템 인덱스 계산
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    if len(customer_table) < end_index:
        end_index = len(customer_table)
    current_page_data = customer_table.iloc[start_index:end_index]

    # customer table 출력
    with st.expander("이탈 고위험 군 고객 리스트"):
        st.dataframe(current_page_data, width=1200, height=390, hide_index=True)

        # 페이지 이동 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if page > 1:
                if st.button("이전"):
                    st.session_state.page -= 1
                    st.rerun(scope="fragment")
        
        with col2:
            st.write(f"{page} - {total_pages}")

        with col3:
            if page < total_pages:
                if st.button("다음"):
                    st.session_state.page += 1
                    st.rerun(scope="fragment")

# 고객 테이블 출력
@st.fragment
def show_customer_table(customer_table):
    if not customer_table.empty:
        show_customer_pages(customer_table=customer_table)
    else:
        st.write("고객 정보를 불러올 수 없습니다.")


################### 분석 결과 구성 ###################
def show_analysis_result(analysis_option):

    # 결과 컨테이너 생성
    analysis_container = st.container()

    if analysis_option == "변수 중요도":
        importance_metric = st.selectbox(
            "변수 중요도 측정 방식 선택",
            ["SHAP value (범주형)",
             "SHAP value (연속형)",
             "CHI-Square / P-value (범주형 변수)"]
        )

        if importance_metric == "SHAP value (범주형)":

            tab1, tab2 = st.tabs(["분석 그래프", "분석 테이블"])

            with tab1:
                # 변수 중요도 이미지 로드
                shap_summary = Image.open(CATEGORICAL_SHAP_SUMMARY_PATH)
                shap_summary_bar = Image.open(CATEGORICAL_SHAP_SUMMARY_BAR_PATH)

                # 이미지 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.image(shap_summary, caption='SHAP value 요약', use_column_width=True)
                
                with col2:
                    st.image(shap_summary_bar, caption='평균 SHAP value 요약 (막대)', use_column_width=True)
            with tab2:
                feature_importance = pd.read_csv(CATEGORICAL_MEAN_SHAP_VALUE_PATH, index_col='Unnamed: 0')
                st.dataframe(feature_importance, width=800, height=600)
            
            st.markdown(f"### SHAP value")
            st.markdown(
                """
                > SHAP value가 클 수록 해당 변수가 이탈 확률에 큰 영향을 미쳤다는 것을 의미합니다.
                #### 요약 그래프 해석
                - 변수의 SHAP value의 분포를 확인할 수 있습니다.
                - 해당 변수가 중요할 수록 높거나 낮은 SHAP value의 분포가 넓게 나타납니다.
                - 특정 변수가 이탈 확률에 큰 영향을 미친 경우 그 변수의 SHAP value가 다른 변수보다 더 넓은 범위를 가집니다.
                """
            )

        
        elif importance_metric == "SHAP value (연속형)":

            tab1, tab2 = st.tabs(["분석 그래프", "분석 테이블"])

            with tab1:
                # 변수 중요도 이미지 로드
                shap_summary = Image.open(NUMERIC_SHAP_SUMMARY_PATH)
                shap_summary_bar = Image.open(NUMERIC_SHAP_SUMMARY_BAR_PATH)

                # 이미지 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.image(shap_summary, caption='SHAP value 요약', use_column_width=True)
                
                with col2:
                    st.image(shap_summary_bar, caption='평균 SHAP value 요약 (막대)', use_column_width=True)
            with tab2:
                feature_importance = pd.read_csv(NUMERIC_MEAN_SHAP_VALUE_PATH, index_col='Unnamed: 0')
                st.dataframe(feature_importance, width=800, height=600)
            
            st.markdown(f"### SHAP value")
            st.markdown(
                """
                > SHAP value가 클 수록 해당 변수가 이탈 확률에 큰 영향을 미쳤다는 것을 의미합니다.
                #### 요약 그래프 해석
                - 변수의 SHAP value의 분포를 확인할 수 있습니다.
                - 해당 변수가 중요할 수록 높거나 낮은 SHAP value의 분포가 넓게 나타납니다.
                - 특정 변수가 이탈 확률에 큰 영향을 미친 경우 그 변수의 SHAP value가 다른 변수보다 더 넓은 범위를 가집니다.
                """
            )
            # st.markdown("---")
        
        elif importance_metric == "CHI-Square / P-value (범주형 변수)":

            tab1, tab2 = st.tabs(["분석 그래프", "분석 테이블"])

            with tab1 :
                feature_importance = pd.read_csv(CHI2_TABLE_PATH, index_col='Unnamed: 0')

                col1, col2 = st.columns(2)
                with col1:
                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance['chi2']))
                    bar_fig.update_layout(barmode='group', 
                                        title="변수 별 CHI-Square 값", 
                                        xaxis_title="변수", 
                                        yaxis_title="CHI-Square")
                    st.plotly_chart(bar_fig)
                
                with col2:
                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(x=feature_importance.index, y=feature_importance['p_value']))
                    bar_fig.update_layout(barmode='group', 
                                        title="변수 별 P-Value 값", 
                                        xaxis_title="변수", 
                                        yaxis_title="P-Value")
                    st.plotly_chart(bar_fig)
            with tab2 :
                feature_importance = pd.read_csv(CHI2_TABLE_PATH, index_col='Unnamed: 0')
                st.dataframe(feature_importance, width=800, height=600)
            
            st.markdown(f"### CHI-Square / P-value")
            st.markdown(
                """
                > CHI Square가 높고, P-Value가 0.05 보다 낮으면 중요한 변수 라는 것을 의미합니다.

                > 범주형 변수에 대해서만 측정이 가능합니다.
                
                #### 요약 그래프 해석
                ##### CHI Square 
                - 값이 클 수록, 해당 변수와 이탈 여부 간의 상관성이 강하다는 것을 의미합니다.
                
                ##### P-Value
                - 일반적으로 P-Value가 0.05보다 작으면 귀무가설을 기각하며, 해당 변수와 종속 변수 간에 유의미한 관계가 있다고 결론 짓습니다.
                - 즉, P-Value가 0.05 이하로 충분히 낮다면 변수의 중요도가 높다는 것을 의미합니다.
                """
            )
    
    elif analysis_option == "특정 변수의 이탈 여부 별 분포":
        feature_option = st.selectbox(
                "분포를 확인할 변수 선택",
                FEATURE_LIST
            )
        
        tab1, tab2 = st.tabs(["분석 그래프", "분석 테이블"])

        total_data = pd.read_csv(PREPROCESSED_DATA_PATH)

        with tab1 :
        
            if feature_option == "Churn":
                labels = ['No', 'Yes']
                values = total_data['Churn'].value_counts()
                pie_fig = go.Figure()
                pie_fig.add_trace(go.Pie(labels=labels, values=values))
                pie_fig.update_layout(title="Churn 분포", )
                st.plotly_chart(pie_fig)
            else:
               
                fig = px.histogram(
                    total_data,
                    x=feature_option,
                    color='Churn',
                    barmode='overlay',
                    histnorm='probability density',  # 확률 밀도 함수로 정규화
                    labels={feature_option: f"{feature_option}", 'Churn': '이탈 여부'},
                    title=f'{feature_option}에 따른 이탈 여부 분포',
                    nbins=50
                )
                st.plotly_chart(fig)
            
        with tab2 :
            if feature_option == "Churn":
                st.dataframe(total_data[feature_option].value_counts(), 
                             width=800, height=110)
            else:
                st.dataframe(total_data.groupby("Churn")[feature_option].describe(), 
                             width=800, height=110)