
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain_core.tools import tool

from dotenv import load_dotenv
import requests
import os
import json
import sqlite3
import time # Diperlukan untuk jeda di tools

conn = sqlite3.connect("data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    message TEXT
)
""")
conn.commit()


def parse_input(input_str):
    """Fungsi pembantu untuk mem-parse string input tool."""
    parts = input_str.split(";")
    return dict(pair.split("=") for pair in parts if "=" in pair)


@tool
def get_brands_str(input_str: str) -> str:
    """
    Get available vehicle brands for a specific vehicle type.
    
    Use this tool to get a list of vehicle brands and their respective codes.
    Input: 'vehicle_type=carros' (options: carros, motos, caminhoes)
    """
    try:
        params = parse_input(input_str)
        vt = params.get("vehicle_type", "carros") 
        limit = int(params.get("limit", 20)) 
        
        if vt not in ["carros", "motos", "caminhoes"]:
             return json.dumps({"error": f"Tipe kendaraan '{vt}' tidak valid. Pilih antara 'carros', 'motos', atau 'caminhoes'."})

        url = f"https://parallelum.com.br/fipe/api/v1/{vt}/marcas"
        data = requests.get(url, timeout=10).json()
        
        html_output = f"<h3>🚗 Merek untuk Tipe Kendaraan: {vt.upper()}</h3>"
        html_output += "<ul>"
        for item in data[:limit]:
            html_output += f"<li><b>{item['nome']}</b> (Kode: <code>{item['codigo']}</code>)</li>"
        html_output += "</ul>"
        html_output += "<i>Silakan pilih kode merek yang Anda minati.</i>"
        
        return html_output
    except Exception as e:
        return json.dumps({"error": f"Kesalahan saat mengambil merek: {str(e)}"})


@tool
def get_models_and_years_tool(input_str: str) -> str:
    """
    Get vehicle models and available years. Requires vehicle_type and brand_code.
    
    Input: 'vehicle_type=carros;brand_code=7;limit=5' (limit is optional, default 5)
    """
    try:
        params = parse_input(input_str)
        vt = params.get("vehicle_type")
        brand_code = params.get("brand_code")
        limit = int(params.get("limit", 5))

        if not vt or not brand_code:
            return json.dumps({"error": "Harap tentukan 'vehicle_type' dan 'brand_code'."})

        models_url = f"https://parallelum.com.br/fipe/api/v1/{vt}/marcas/{brand_code}/modelos"
        models_data = requests.get(models_url, timeout=10).json()
        modelos = models_data.get("modelos", [])[:limit]

        if not modelos:
            return f"Tidak ada model ditemukan untuk kode merek {brand_code} (Tipe: {vt})."

        html_blocks = []

        for model in modelos:
            model_name = model['nome']
            model_code = model['codigo']

            years_url = f"https://parallelum.com.br/fipe/api/v1/{vt}/marcas/{brand_code}/modelos/{model_code}/anos"
            years_data = requests.get(years_url, timeout=10).json()[:5] 

            years_display = ", ".join([f"{y['nome']} (<code>{y['codigo']}</code>)" for y in years_data])
            
            html_blocks.append(
              f"""
              <div style="border-left: 3px solid #00BCD4; padding: 10px; margin-bottom: 8px; background-color: #f0f8ff; border-radius: 4px;">
                  <b>🚗 Model: {model_name}</b> (Kode Model: <code>{model_code}</code>)<br>
                  📅 Tahun Tersedia: {years_display if years_display else 'Tidak ada data tahun.'}
              </div>
              """
            )
            time.sleep(0.1) 

        return "\n".join(html_blocks)

    except Exception as e:
        return json.dumps({"error": f"Kesalahan saat mengambil model/tahun: {str(e)}. Pastikan vehicle_type dan brand_code sudah benar."})


def build_agent():
    load_dotenv()
    
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise ValueError("REPLICATE_API_TOKEN tidak ditemukan di environment variable. Pastikan Anda mengaturnya di Google Colab secrets.")

    llm = Replicate(model="anthropic/claude-3.5-haiku")

    system_message = """
    Anda adalah **Asisten Rekomendasi Kendaraan Profesional dan Ramah**.
    Tugas Anda adalah membantu pengguna menemukan mobil, motor, atau truk yang paling sesuai dengan kebutuhan dan preferensi mereka.

    **PRINSIP KERJA:**
    1.  **Sapaan dan Pengantar:** Selalu sapa pengguna dengan ramah dan tanyakan kebutuhan mereka.
    2.  **Kumpulkan Preferensi:** Tanyakan kepada pengguna tentang Tipe Kendaraan (gunakan kode FIPE: `carros`, `motos`, atau `caminhoes`).
    3.  **Gunakan Perkakas:** Gunakan `get_brands_str` untuk mendapatkan daftar merek, lalu `get_models_and_years_tool` setelah pengguna memilih kode merek.
    4.  **Berikan Rekomendasi Akhir:** Berikan rekomendasi berupa **daftar mobil yang sesuai** berdasarkan data yang dikumpulkan.
    """

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    tools = [
        get_brands_str,
        get_models_and_years_tool,
    ]

    agent_executor = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )

    return agent_executor
