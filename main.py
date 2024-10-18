import streamlit as st
from project_package.StreamlitModule import fetch
from project_package.StreamlitModule import show_analysis_result, show_customer_table



######## Streamlit 앱 시작 ########
st.set_page_config(page_title="Customer Analysis", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center'>고객 이탈 관리 프로그램</h1>
    <br>
    <h6 style='display: inline; float: right;'>1팀:  서장호, 박찬규, 배윤관, 최영민</h6>
    <br> 
    """,
    unsafe_allow_html=True
)

fetch_result = fetch()

customer_table = fetch_result['customer_table']
model = fetch_result['model']



with st.container():
    st.subheader("이탈 고위험 고객")

    show_customer_table(customer_table=customer_table)

st.markdown('---')


with st.container():
    st.subheader("위험성 분석")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('#### 변수 중요도')

        show_analysis_result(analysis_option="변수 중요도")
    
    with col2:
        st.markdown('#### 특정 변수의 이탈 여부 별 분포')

        show_analysis_result(analysis_option="특정 변수의 이탈 여부 별 분포")

st.markdown('---')