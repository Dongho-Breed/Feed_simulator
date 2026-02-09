import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 원료 데이터 설정 ---
feed_data = {
    '원료명': ['알팔파(조사료)', 'IRG 사료(조사료)', '볏짚(조사료)', '옥수수(농후)', '배합사료(농후)', 'TMR'],
    'TDN': [52.5, 37.6, 39.0, 76.7, 70.0, 68.0],
    'CP': [19.8, 6.4, 4.5, 7.2, 17.0, 14.0],
    'NDF': [49.9, 33.8, 70.0, 8.4, 27.0, 32.0]
}
df_feed = pd.DataFrame(feed_data)

# --- 2. 단계별 가변 데이터 (엑셀 데이터 기반) ---
stage_specs = {
    "비육우 육성기(6~12)": {
        "title_eng": "Growing Stage (6-12m)",
        "target_tdn": 69.0, "target_cp": 15.0, "min_ndf": 30.0, 
        "weight": 234.0, "weight_gain": 0.027, "days": 180, "dmi": 6.318,
        "default_ratios": [23.2, 0.0, 21.2, 0.0, 55.6, 0.0]
    },
    "비육기 전기(13~18)": {
        "title_eng": "Early Fattening (13-18m)",
        "target_tdn": 71.0, "target_cp": 11.5, "min_ndf": 28.0, 
        "weight": 375.0, "weight_gain": 0.028, "days": 180, "dmi": 10.5,
        "default_ratios": [9.5, 33.7, 6.3, 25.3, 16.8, 8.4]
    },
    "비육기 후기(19~30)": {
        "title_eng": "Late Fattening (19-30m)",
        "target_tdn": 72.5, "target_cp": 10.5, "min_ndf": 25.0, 
        "weight": 517.0, "weight_gain": 0.024, "days": 334, "dmi": 12.408,
        "default_ratios": [2.0, 0.0, 3.0, 23.8, 71.2, 0.0]
    }
}

st.set_page_config(page_title="한우 정밀 영양 시뮬레이터", layout="wide")
st.title("🐂 한우 단계별 정밀 영양 시뮬레이션")

# --- 3. 사이드바 설정 (가변 값 자동 전환) ---
selected_stage = st.sidebar.selectbox("사양 단계를 선택하세요", list(stage_specs.keys()))
spec = stage_specs[selected_stage]

st.sidebar.divider()
st.sidebar.header("🟦 사양 관리 설정 (가변)")
u_days = st.sidebar.number_input("육성 일수 (일)", value=spec['days'], key=f"days_{selected_stage}")
u_weight = st.sidebar.number_input("평균 체중 (kg)", value=spec['weight'], key=f"weight_{selected_stage}")
u_gain = st.sidebar.number_input("체중비", value=spec['weight_gain'], format="%.3f", key=f"gain_{selected_stage}")
st.sidebar.info(f"일일 DMI: {spec['dmi']} kg (고정)")

st.sidebar.divider()
st.sidebar.header("🟦 사료 배합 비율 (%)")
user_ratios = []
for i, name in enumerate(df_feed['원료명']):
    val = st.sidebar.number_input(f"{name}", min_value=0.0, max_value=100.0, value=spec['default_ratios'][i], step=0.1, key=f"f_{i}_{selected_stage}")
    user_ratios.append(val)

# --- 4. 영양소 계산 및 판정 (수치 원복) ---
mixed_tdn = sum([r * t / 100 for r, t in zip(user_ratios, df_feed['TDN'])])
mixed_cp = sum([r * c / 100 for r, c in zip(user_ratios, df_feed['CP'])])
mixed_ndf = sum([r * n / 100 for r, n in zip(user_ratios, df_feed['NDF'])])

tdn_ok = "✅ OK" if mixed_tdn >= spec['target_tdn'] else "❌ 부족"
cp_ok = "✅ OK" if mixed_cp >= spec['target_cp'] else "❌ 부족"
ndf_ok = "✅ OK" if mixed_ndf >= spec['min_ndf'] else "❌ 부족"

# --- 5. 상단 지표 대시보드 ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("육성 일수", f"{u_days} 일")
m2.metric("평균 체중", f"{u_weight} kg")
m3.metric("체중비", f"{u_gain}")
m4.metric("일일 DMI", f"{spec['dmi']} kg")

st.divider()

# --- 6. 영양소 판정 결과 ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("혼합 TDN", f"{mixed_tdn:.2f}%", f"목표: {spec['target_tdn']}%")
    st.subheader(f"판정: {tdn_ok}")
with c2:
    st.metric("혼합 CP", f"{mixed_cp:.2f}%", f"목표: {spec['target_cp']}%")
    st.subheader(f"판정: {cp_ok}")
with c3:
    st.metric("혼합 NDF", f"{mixed_ndf:.2f}%", f"하한: {spec['min_ndf']}%")
    st.subheader(f"판정: {ndf_ok}")

# --- 7. 파이 차트 (네모박스 제거 및 제목 클리닝) ---
st.divider()
st.write("### 📋 현재 사료 배합 비율 분석")
col_l, col_r = st.columns([1, 2])

with col_l:
    for name, ratio in zip(df_feed['원료명'], user_ratios):
        if ratio > 0:
            st.write(f"- {name}: **{ratio}%**")

with col_r:
    plot_ratios = [r for r in user_ratios if r > 0]
    eng_labels = ['Alfalfa', 'IRG', 'Straw', 'Corn', 'Concentrate', 'TMR']
    plot_labels = [eng_labels[i] for i, r in enumerate(user_ratios) if r > 0]
    
    if sum(plot_ratios) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        # 파이 차트 제목에서 한글을 제거하여 네모박스 방지
        ax.pie(
            plot_ratios, 
            labels=plot_labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            pctdistance=0.85, 
            labeldistance=1.1,
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'],
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        # 차트 제목을 영문으로만 설정하여 깔끔하게 표시
        ax.set_title(f"Feed Composition: {spec['title_eng']}", fontsize=16, pad=20)
        st.pyplot(fig)
