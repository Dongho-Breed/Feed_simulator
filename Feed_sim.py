import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 원료 데이터 설정 ---
feed_data = {
    '원료명': ['알팔파(조사료)', 'IRG 사료(조사료)', '볏짚(조사료)', '옥수수(농후)', '배합사료(농후)', 'TMR'],
    '단가': [900, 350, 200, 550, 650, 600],
    'TDN': [52.5, 37.6, 39.0, 76.7, 70.0, 68.0],
    'CP': [19.8, 6.4, 4.5, 7.2, 17.0, 14.0],
    'NDF': [49.9, 33.8, 70.0, 8.4, 27.0, 32.0]
}
df_feed = pd.DataFrame(feed_data)

# --- 2. 단계별 가변 데이터 매핑 (엑셀 파란색 셀 수치 반영) ---
stage_specs = {
    "비육우 육성기(6~12)": {
        "target_tdn": 69.0, "target_cp": 15.0, "min_ndf": 30.0, 
        "weight": 234.0, "weight_gain": 0.027, "days": 180, "dmi": 6.318,
        "default_ratios": [23.2, 0.0, 21.2, 0.0, 55.6, 0.0]
    },
    "비육기 전기(13~18)": {
        "target_tdn": 71.0, "target_cp": 11.5, "min_ndf": 28.0, 
        "weight": 375.0, "weight_gain": 0.028, "days": 180, "dmi": 10.5,
        "default_ratios": [9.5, 33.7, 6.3, 25.3, 16.8, 8.4]
    },
    "비육기 후기(19~30)": {
        "target_tdn": 72.5, "target_cp": 10.5, "min_ndf": 25.0, 
        "weight": 517.0, "weight_gain": 0.024, "days": 334, "dmi": 12.408,
        "default_ratios": [2.0, 0.0, 3.0, 23.8, 71.2, 0.0]
    }
}

st.set_page_config(page_title="한우 정밀 사양 시뮬레이터", layout="wide")
st.title("🐂 한우 단계별 가변 사양 시뮬레이션")

# --- 3. 사이드바: 가변 데이터 입력 ---
selected_stage = st.sidebar.selectbox("사양 단계를 선택하세요", list(stage_specs.keys()))
spec = stage_specs[selected_stage]

st.sidebar.divider()
st.sidebar.header("🟦 사양 관리 설정 (가변)")
# 단계 선택 시 자동으로 값이 바뀌도록 key를 selected_stage와 연동
u_days = st.sidebar.number_input("육성 일수 (일)", value=spec['days'], key=f"days_{selected_stage}")
u_weight = st.sidebar.number_input("평균 체중 (kg)", value=spec['weight'], key=f"weight_{selected_stage}")
u_gain = st.sidebar.number_input("체중비", value=spec['weight_gain'], format="%.3f", key=f"gain_{selected_stage}")
# DMI는 고정값으로 두되 표시만 함
st.sidebar.info(f"일일 DMI: {spec['dmi']} kg (고정)")

st.sidebar.divider()
st.sidebar.header("🟦 사료 배합 비율 (%)")
user_ratios = []
for i, name in enumerate(df_feed['원료명']):
    val = st.sidebar.number_input(f"{name}", min_value=0.0, max_value=100.0, value=spec['default_ratios'][i], step=0.1, key=f"f_{i}_{selected_stage}")
    user_ratios.append(val)

# --- 4. 영양소 계산 및 판정 ---
mixed_tdn = sum([r * t / 100 for r, t in zip(user_ratios, df_feed['TDN'])])
mixed_cp = sum([r * c / 100 for r, c in zip(user_ratios, df_feed['CP'])])
mixed_ndf = sum([r * n / 100 for r, n in zip(user_ratios, df_feed['NDF'])])

tdn_ok = "✅ OK" if mixed_tdn >= spec['target_tdn'] else "❌ 부족"
cp_ok = "✅ OK" if mixed_cp >= spec['target_cp'] else "❌ 부족"
ndf_ok = "✅ OK" if mixed_ndf >= spec['min_ndf'] else "❌ 부족"

# --- 5. 대시보드 출력 (바뀌는 값 강조) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("선택된 육성 일수", f"{u_days} 일")
m2.metric("현재 평균 체중", f"{u_weight} kg")
m3.metric("현재 체중비", f"{u_gain}")
m4.metric("일일 DMI", f"{spec['dmi']} kg")

st.divider()

# 영양소 판정 결과
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

# --- 6. 파이 차트 (TMR 포함 비율) ---
st.divider()
col_l, col_r = st.columns([1, 1.5])
with col_l:
    st.write("### 📋 현재 배합 리포트")
    for name, ratio in zip(df_feed['원료명'], user_ratios):
        if ratio > 0:
            st.write(f"- {name}: **{ratio}%**")

with col_r:
    plot_ratios = [r for r in user_ratios if r > 0]
    plot_labels = [df_feed['원료명'][i].split('(')[0] for i, r in enumerate(user_ratios) if r > 0]
    
    if sum(plot_ratios) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(plot_ratios, labels=plot_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'])
        ax.set_title(f"{selected_stage} Composition", fontsize=14)
        st.pyplot(fig)

# --- 7. 경제성 분석 (가변 데이터 u_days, u_weight 반영) ---
st.divider()
avg_price = sum([r * p / 100 for r, p in zip(user_ratios, df_feed['단가'])])
daily_cost = avg_price * spec['dmi']
total_feed_cost = daily_cost * u_days

st.subheader("📊 사료 경제성 분석")
g1, g2 = st.columns(2)
with g1:
    st.info(f"💰 일일 사료비: {int(daily_cost):,} 원")
    st.success(f"💰 {u_days}일간 총 사료비: {int(total_feed_cost):,} 원")
with g2:
    # 수익 지표: (가상 매출 - 총사료비) / 가중치
    profit_index = (15000 * u_weight - total_feed_cost) / 10000 
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
    ax_bar.bar(['Profit Index'], [profit_index], color='#27ae60')
    ax_bar.set_title("Economic Indicators based on Weight & Days")
    st.pyplot(fig_bar)