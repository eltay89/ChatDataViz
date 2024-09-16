import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import matplotlib.pyplot as plt
import io

class LMStudioLLM:
  def __init__(self, api_url: str = "http://127.0.0.1:1234", model_name: str = "Qwen/Qwen2-1_5b-instruct-q8_0/qwen2-1_5b-instruct-q8_0.gguf"):
      self.api_url = api_url
      self.model_name = model_name

  def generate(self, prompt: str, **kwargs) -> str:
      headers = {"Content-Type": "application/json"}
      data = {
          "model": self.model_name,
          "messages": [{"role": "user", "content": prompt}],
          "temperature": kwargs.get('temperature', 0.7),
          "max_tokens": kwargs.get('max_tokens', -1),
          "stream": False
      }
      try:
          response = requests.post(f"{self.api_url}/v1/chat/completions", headers=headers, json=data)
          response.raise_for_status()
          return response.json()['choices'][0]['message']['content'].strip()
      except requests.exceptions.RequestException as e:
          st.error(f"Error communicating with LM Studio: {e}")
          return ""

  def analyze_data(self, df: pd.DataFrame, question: str) -> str:
      data_info = df.head(10).to_string()
      columns_info = df.dtypes.to_string()
      basic_stats = df.describe().to_string()
      prompt = f"""Given the following dataset information:

Column names and types:
{columns_info}

Data sample (first 10 rows):
{data_info}

Basic statistics:
{basic_stats}

Please answer the following question about the data:
{question}

Provide a concise and accurate answer based on the information given. If you cannot answer the question with the given information, please say so and explain why."""
      return self.generate(prompt)

  def generate_plot(self, df: pd.DataFrame, question: str):
      prompt = f"""Given the following data columns: {', '.join(df.columns)}
      Create a Python script using matplotlib to plot a chart answering this question: {question}
      Use only matplotlib.pyplot as plt and pandas as pd.
      Return only the Python code, no explanations or comments.
      The code should start with 'import matplotlib.pyplot as plt' and end with 'plt.tight_layout()'."""
      code = self.generate(prompt)
      try:
          exec(code)
          buf = io.BytesIO()
          plt.savefig(buf, format='png')
          buf.seek(0)
          plt.close()
          return buf
      except Exception as e:
          return f"Failed to generate chart: {str(e)}\nGenerated code:\n{code}"

@st.cache_data
def load_data(file):
  try:
      if file.name.endswith('.csv'):
          df = pd.read_csv(file)
          return {"Sheet1": preprocess_data(df)}
      elif file.name.endswith(('.xls', '.xlsx')):
          xls = pd.ExcelFile(file)
          return {sheet_name: preprocess_data(pd.read_excel(xls, sheet_name)) for sheet_name in xls.sheet_names}
      else:
          st.error("Unsupported file format. Please upload a CSV or Excel file.")
          return None
  except Exception as e:
      st.error(f"An error occurred while loading the data: {e}")
      return None

def preprocess_data(df):
  df.columns = df.columns.str.lower().str.replace(' ', '_')
  for col in df.select_dtypes(include=['object']):
      df[col] = df[col].str.strip()
  for col in df.columns:
      if 'date' in col:
          try:
              df[col] = pd.to_datetime(df[col])
          except:
              pass
  return df

def plot_data(df, x_col=None, y_col=None, plot_type='line'):
  try:
      if plot_type == 'line':
          fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
      elif plot_type == 'scatter':
          fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
      elif plot_type == 'bar':
          fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
      elif plot_type == 'histogram':
          fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
      elif plot_type == 'heatmap':
          fig = px.imshow(df.corr(), text_auto=True, title="Correlation Heatmap")
      elif plot_type == 'box':
          fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
      else:
          st.error(f"Plot type '{plot_type}' not recognized.")
          return
      st.plotly_chart(fig, use_container_width=True)
  except Exception as e:
      st.error(f"An error occurred while plotting: {e}")

def main():
  st.set_page_config(page_title="ChatDataViz", layout="wide")

  # Set light grey background
  st.markdown(
      """
      <style>
      .stApp {
          background-color: #f0f0f0;
      }
      </style>
      """,
      unsafe_allow_html=True
  )

  if 'messages' not in st.session_state:
      st.session_state.messages = []
  if 'data' not in st.session_state:
      st.session_state.data = None
  if 'current_sheet' not in st.session_state:
      st.session_state.current_sheet = None

  st.title("🤖💬 The AI-Powered Data Analysis Companion")

  llm = LMStudioLLM()

  col1, col2 = st.columns([3, 2])

  with col1:
      st.markdown("---")
      st.header("💬 Chat & Data Analysis")
      st.markdown("---")

      # File upload
      uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
      if uploaded_file:
          data = load_data(uploaded_file)
          if data:
              st.session_state.data = data
              st.success("✅ Data loaded successfully!")
              
              # Sheet selection
              sheet_names = list(data.keys())
              st.session_state.current_sheet = st.selectbox("Select a sheet:", sheet_names)
              
              st.markdown("### Data Preview:")
              st.dataframe(data[st.session_state.current_sheet].head(), use_container_width=True)
      
      st.markdown("---")
      # Chat interface
      st.subheader("Chat with AI")
      st.markdown("---")
      for msg in st.session_state.messages:
          with st.chat_message(msg["role"]):
              st.markdown(msg["content"])

      if prompt := st.chat_input("Ask a question or request data analysis..."):
          st.session_state.messages.append({"role": "user", "content": prompt})
          with st.chat_message("user"):
              st.markdown(prompt)

          with st.chat_message("assistant"):
              if st.session_state.data and st.session_state.current_sheet:
                  response = llm.analyze_data(st.session_state.data[st.session_state.current_sheet], prompt)
              else:
                  response = llm.generate(prompt)
              st.markdown(response)
          st.session_state.messages.append({"role": "assistant", "content": response})

      # Clear button
      if st.button("🧹 Clear Chat"):
          st.session_state.messages = []
          st.experimental_rerun()

  with col2:
      st.markdown("---")
      st.header("📊 Data Visualization")
      st.markdown("---")

      if st.session_state.data and st.session_state.current_sheet:
          df = st.session_state.data[st.session_state.current_sheet]

          st.markdown("### Generate Custom Visualization")
          st.markdown("---")
          viz_question = st.text_input("Describe the visualization you want:")
          if st.button("🎨 Generate Custom Plot"):
              with st.spinner("Generating visualization..."):
                  chart = llm.generate_plot(df, viz_question)
                  if isinstance(chart, io.BytesIO):
                      st.image(chart)
                  else:
                      st.error(chart)

          st.markdown("---")
          st.markdown("### Quick Plot")
          st.markdown("---")
          plot_type = st.selectbox("Select plot type:", ["line", "scatter", "bar", "histogram", "heatmap", "box"])
          if plot_type != "heatmap":
              columns = df.columns.tolist()
              x_col = st.selectbox("Select X-axis:", options=columns)
              y_col = st.selectbox("Select Y-axis:", options=columns)
          else:
              x_col = y_col = None

          if st.button("📊 Generate Quick Plot"):
              if plot_type != "heatmap":
                  if x_col and y_col:
                      plot_data(df, x_col, y_col, plot_type)
                  else:
                      st.error("Please select both X-axis and Y-axis columns.")
              else:
                  plot_data(df, plot_type=plot_type)
      else:
          st.info("📂 Please upload a CSV or Excel file to enable data visualization.")

  st.markdown("---")
  st.markdown("Created with ❤️ by Mohamed")

if __name__ == '__main__':
  main()