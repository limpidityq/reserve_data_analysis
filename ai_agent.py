import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from openai import OpenAI
import httpx
import re
import os

# 1. 配置客户端
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    http_client=httpx.Client(timeout=180.0)
)

def get_parquet_path(file_path):
    base, _ = os.path.splitext(file_path)
    return f"{base}_cache.parquet"

# --- 核心改进：传入 chat_history ---
def actuary_file_agent(user_query, file_path, chat_history):
    if not os.path.exists(file_path):
        return f"❌ 路径错误：找不到文件 {file_path}"

    conn = duckdb.connect(database=':memory:')
    ext = os.path.splitext(file_path)[-1].lower()
    
    parquet_path = get_parquet_path(file_path)
    if ext == '.csv':
        if not os.path.exists(parquet_path):
            conn.execute(f"COPY (SELECT * FROM read_csv_auto('{file_path}')) TO '{parquet_path}' (FORMAT PARQUET)")
        target_source = parquet_path
    else:
        target_source = file_path

    conn.execute(f"CREATE VIEW claims_table AS SELECT * FROM read_parquet('{target_source}')")
    real_cols = conn.execute("SELECT * FROM claims_table LIMIT 0").df().columns.tolist()
    columns_info = ", ".join(real_cols)

    # 构造上下文摘要（只取最近3轮，防止Token溢出）
    history_context = ""
    if chat_history:
        history_context = "### 历史对话上下文 ###\n"
        for turn in chat_history[-3:]:
            history_context += f"用户: {turn['user']}\n助手结论: {turn['assistant']}\n"

    # SQL 模型指令：强化对“追问”的理解
    sql_prompt = f"""
    {history_context}
    
    当前表: claims_table, 字段: {columns_info}
    最新任务：{user_query}
    
    输出要求：
    1. 如果任务是针对前序结果的追问（如“那刚才提到的车型里...”），请参考上下文编写 SQL。
    2. SQL代码块：```sql ... ```
    3. 别名若以数字开头请加双引号。
    """

    try:
        response = client.chat.completions.create(
            model="actuary-gpt",
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0
        )
        full_content = response.choices[0].message.content
        
        sql_match = re.search(r"```sql\s*(.*?)\s*```", full_content, re.DOTALL | re.IGNORECASE)
        if not sql_match:
            return "❌ AI 未能生成 SQL，请尝试换种问法。"
        
        sql_query = sql_match.group(1).strip()
        
        # 字段自动纠偏
        corrections = {"再保前金额": "再保前赔款", "claim_amount": "paid_amount"}
        for wrong, right in corrections.items():
            if right in columns_info: sql_query = sql_query.replace(wrong, right)
        sql_query = re.sub(r"AS\s+([0-9]\S*)", r'AS "\1"', sql_query)

        print(f"\n--- 执行 SQL ---\n{sql_query}")
        result_df = conn.execute(sql_query).df()
        print(result_df)

        # 绘图逻辑
        plot_match = re.search(r"\[PLOT:\s*(.*?)\]", full_content, re.IGNORECASE)
        if plot_match and not result_df.empty:
            draw_plot(result_df, plot_match.group(1).lower().strip(), user_query)

        # 解读模型指令：结合上下文提供连贯结论
        interpretation_prompt = f"""
        {history_context}
        用户最新需求: {user_query}
        最新查询结果:
        {result_df.to_markdown(index=False)}
        
        请作为资深精算师点评。如果是追问，请保持逻辑连贯。
        """
        
        analysis_res = client.chat.completions.create(
            model="analysis-gpt",
            messages=[{"role": "user", "content": interpretation_prompt}],
            temperature=0.7
        )
        
        final_analysis = analysis_res.choices[0].message.content
        print("\n📊 --- 精算深度解读 ---")
        print(final_analysis)

        # --- 记忆存储 ---
        chat_history.append({"user": user_query, "assistant": final_analysis})
        
        return "SUCCESS"

    except Exception as e:
        return f"❌ 运行报错: {str(e)}"

# draw_plot 保持不变...
def draw_plot(df, kind, title):
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if kind == 'development_compare' or ('accident_year' in df.columns and 'development_month' in df.columns):
            df_pivot = df.pivot(index='development_month', columns='accident_year', values=df.columns[-1])
            df_pivot.plot(kind='line', marker='o', figsize=(10, 6))
            plt.title(f"精算进展对比：{title}")
            plt.show()
        elif kind == 'dual_axis' or (df.shape[1] >= 3 and ('比例' in df.columns[-1] or 'ratio' in df.columns[-1].lower())):
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.bar(df.iloc[:, 0], df.iloc[:, 1], color='steelblue', alpha=0.6)
            ax2 = ax1.twinx()
            ax2.plot(df.iloc[:, 0], df.iloc[:, -1], color='darkorange', marker='D')
            plt.title(title)
            plt.show()
        else:
            df.set_index(df.columns[0]).plot(kind=kind if kind in ['line','bar','pie'] else 'bar', figsize=(10, 5), title=title)
            plt.show()
    except Exception as e:
        print(f"绘图异常: {e}")

# --- 交互式主入口 ---
if __name__ == "__main__":
    path = input("请输入数据文件路径 (CSV/Parquet): ").strip('"')
    # 初始化本轮会话的记忆
    session_history = []
    
    print("\n🚀 助手已就绪！输入 'exit' 或 'quit' 退出，输入 'clear' 清空记忆。")
    
    while True:
        query = input("\n👤 用户需求: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            break
        if query.lower() == 'clear':
            session_history = []
            print("✨ 记忆已清空")
            continue
        if not query:
            continue
            
        status = actuary_file_agent(query, file_path=path, chat_history=session_history)
        if status != "SUCCESS":
            print(status)