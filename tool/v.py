import streamlit as st
import pandas as pd
import ast
import re
import json
from http import client
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types
from tqdm import tqdm
from utils import *
import pandas as pd
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import anthropic
from openai import OpenAI


llm_chatgpt = None
llm_openai_batch = None
llm_gemini = None
llm_claude = None

GEMINI_CATEGORY_MODEL = "gemini-3.1-pro-preview"
GEMINI_CLASSIFY_MODEL = "gemini-3-flash-preview"
CLAUDE_CATEGORY_MODEL = "claude-opus-4-6"
GEMINI_WORKERS = 20
CLAUDE_MAX_TOKENS = 4096

# Selected models (updated at runtime via seleccionar_llm)
SELECTED_CHATGPT_MODEL = "gpt-5.4-mini"
SELECTED_GEMINI_MODEL = "gemini-3-flash-preview"
SELECTED_CLAUDE_MODEL = "claude-sonnet-4-6"
MIN_APPEND_KEY_OVERLAP = 0.90

def seleccionar_llm_st():
    st.markdown("### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        proveedor = st.selectbox("Provider", ["ChatGPT", "Gemini", "Claude"], index=0)

    API_KEY = st.text_input("Enter your API Key", type="default")
    
    with col2:
        if proveedor == "ChatGPT":
            modelos = gpt_models
            # El recomendado suele ser el primero o uno específico
            modelo_elegido = st.selectbox("ChatGPT Model", modelos, index=0)
            return {"proveedor": "chatgpt", "modelo": modelo_elegido}, API_KEY
            
        elif proveedor == "Gemini":
            modelos = gemini_models
            modelo_elegido = st.selectbox("Gemini Model", modelos, index=0)
            return {"proveedor": "gemini", "modelo": modelo_elegido}, API_KEY
        
        elif proveedor == "Claude":
            modelos = claude_models
            modelo_elegido = st.selectbox("Claude Model", modelos, index=0)
            return {"proveedor": "claude", "modelo": modelo_elegido}, API_KEY

    return None

def input_prompt_component(label, help_text, example_text):
    # We create a visual container for each section
    with st.container(border=True):
        col_tit, col_help = st.columns([0.8, 0.2])
        
        with col_tit:
            st.markdown(f"### {label}")
            
        with col_help:
            # We replace the button/info with a Popover (floating window)
            with st.popover("💡 HELP"):
                st.markdown(f"**Example of {label}:**")
                st.caption(example_text) # More legible text for examples

        # Method selector
        method = st.radio(
            f"How would you like to enter the {label}?",
            ["Write text", "Upload .txt file"],
            key=f"radio_{label}",
            horizontal=True
        )
        
        if method == "Write text":
            return st.text_area(
                f"Enter the {label}:", 
                placeholder=help_text, 
                key=f"txt_{label}",
                height=150 # Fixed height to prevent it from looking too small
            )
        else:
            file = st.file_uploader(
                f"Upload the file for {label}", 
                type="txt", 
                key=f"file_{label}"
            )
            if file:
                return file.read().decode("utf-8")
    return ""

def menu_st(df, menu_type):
    
    estrategia = None
    
    if menu_type == 2:
        st.markdown("---")
        st.subheader("Assignment Strategy")

        # Selección de estrategia
        estrategia = st.radio(
            "Select the prompting strategy::",
            ["Zero-Shot", "Few-Shot", "Zero-Shot CoT", "Few-Shot CoT"],
            horizontal=True
        )
        
    elif menu_type == 1:
        st.markdown("---")
        st.subheader("Categories Creation Strategy")

    info_llm, API_KEY = seleccionar_llm_st()

    if info_llm and API_KEY:
        # Always show pending batches, regardless of whether a file is uploaded

        crear_prompt_obtener_resultados(info_llm, df, API_KEY, estrategia)
        
        mostrar_estado_batches_st(API_KEY)
        
def crear_prompt_obtener_resultados(info_llm, df, API_KEY, estrategia):
    
    # 1. Inicializar estados para persistencia
    if 'proceso_finalizado' not in st.session_state:
        st.session_state.proceso_finalizado = False
    
    st.write(f" **Configured:** {info_llm['proveedor']} ({info_llm['modelo']})")
    st.header("Prompt Configuration")

    # --- BLOQUES BÁSICOS ---
    rol = input_prompt_component("Role", "e.g., You are an economics expert...", HELP_ROLE)
    contexto = input_prompt_component("Context", "e.g., This data comes from...", HELP_CONTEXTO)
    clasificacion = input_prompt_component("Classification", "e.g., Classify into A, B, or C...", HELP_CLASIFICACION)
    formato = input_prompt_component("Format", "e.g., Return a JSON...", HELP_FORMAT)
    constraints = input_prompt_component("Constraints", "e.g., Do not use adjectives...", HELP_CONSTRAINTS)

    extra_content = ""


    # --- BLOQUES DINÁMICOS (Corregidos) ---
    # Usamos 'in' correctamente para detectar la estrategia
    # si estrategia no es None
    if estrategia:
        if "Few-Shot" in estrategia:
            help_text = HELP_FS_COT if "CoT" in estrategia else HELP_FS
            extra_content += "\n" + input_prompt_component("Examples", "Add examples of input/output...", help_text)
        
        # Si la estrategia es "Zero-Shot CoT" o "Few-Shot CoT"
        if "CoT" in estrategia:
            extra_content += "\n" + input_prompt_component("Chain of Thought (CoT)", "Reasoning instructions...", HELP_COT)
        
        
        
    # --- CONFIGURACIÓN EXTRA ---
    st.info("Columns Configuration")
    message_col = st.selectbox(
        "Select the column containing the messages/texts:",
        options=df.columns,
        help="This is the column the LLM will read for classification."
    )
    
    if estrategia:
        game_col_options = ["(none)"] + list(df.columns)
        game_col_sel = st.selectbox(
            "Select the game column (optional):",
            options=game_col_options,
            help="If your data has a game column, the LLM will receive that context."
        )
        game_col = None if game_col_sel == "(none)" else game_col_sel

        keep_cols = st.multiselect(
            "Columns to keep in output (for merging):",
            options=[c for c in df.columns if c != message_col],
            help="These columns will be copied from the original dataset into each result row so you can merge later."
        )

    st.info("Temperatures Configuration")
    temps = configuracion_temperaturas()

    # --- PROCESSING MODE (only for Claude / GPT) ---
    proveedor = info_llm['proveedor']
    if proveedor in ("claude", "chatgpt"):
        st.info("Processing Mode")
        processing_mode = st.radio(
            "How should the LLM process the rows?",
            ["Normal (line by line)", "Batch API (async, ~50% cheaper, up to 24 h)"],
            key="processing_mode_radio",
            horizontal=True,
        )
        if "Batch" in processing_mode:
            if proveedor == "claude":
                st.caption("📊 Track progress: https://console.anthropic.com/settings/workspaces/default/batches")
            else:
                st.caption("📊 Track progress: https://platform.openai.com/batches")
    else:
        processing_mode = "Normal (line by line)"

    # --- CONSTRUCCIÓN DEL PROMPT ---
    partes = [rol, contexto, clasificacion, formato, constraints, extra_content]
    prompt_final = "\n".join([p for p in partes if p.strip()])
    
    strategy_folder = estrategia.lower().replace(" ", "_") if estrategia else "default"

    # --- VALIDACIÓN DE ARCHIVOS PREVIOS (Antes del botón) ---
    # --- VALIDACIÓN DE ARCHIVOS PREVIOS ---

    modos_ejecucion = {}
    hay_archivos_previos = False

    for temp in temps:
        out_file = f"results_line_temp{temp}.csv"
        output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
        
        if os.path.exists(output_path):
            hay_archivos_previos = True
            st.warning(f"⚠️ Previous results detected for Temp: {temp}")
            
            opcion = st.radio(
                f"Action for {out_file}:",
                ["Maintain (Append)", "Delete and Start Fresh (Overwrite)"],
                key=f"radio_{strategy_folder}_{temp}"
            )
            modos_ejecucion[temp] = "overwrite" if "Eliminar" in opcion else "append"
        else:
            modos_ejecucion[temp] = "new"

    # --- VALIDACIÓN FINAL PARA MOSTRAR EL BOTÓN ---
    confirmado = True
    if hay_archivos_previos:
        # Añadimos un checkbox de confirmación final para "frenar" el proceso
        confirmado = st.checkbox("I confirm the file management actions above.", value=False)

    # --- BOTÓN DE EJECUCIÓN (Solo si está confirmado) ---
    if confirmado:
            # --- BOTONES DE EJECUCIÓN Y STOP ---

        if st.button("Generate Prompt and Run"):
            if prompt_final.strip():
                # 1. Aplicar limpieza de archivos
                for temp, modo in modos_ejecucion.items():
                    if modo == "overwrite":
                        out_file = f"results_line_temp{temp}.csv"
                        output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
                        if os.path.exists(output_path):
                            os.remove(output_path)
                
                # 2. Asegurar directorios
                os.makedirs(os.path.join(RESULTS_PATH, proveedor, strategy_folder), exist_ok=True)

                st.session_state.proceso_finalizado = True
                st.rerun() # Forzamos recarga para que entre en el bloque de procesamiento
            else:
                st.error("The prompt is empty.")
    else:
        st.info("Please confirm the file management actions to proceed. If you want to change your selections, please adjust the options above and then check the confirmation box.")

    # --- MOSTRAR RESULTADOS Y PROCESAMIENTO ---
    # si partes no esta empty y el proceso se ha marcado como finalizado, mostramos el prompt y ejecutamos
    if st.session_state.proceso_finalizado:
        # Opcional: Botón para resetear y volver a configurar
        if st.button("STOP, reset, and Edit Prompt"):
            st.session_state.proceso_finalizado = False
            st.rerun()
            
        st.subheader("Final Generated Prompt")
        st.code(prompt_final, language="markdown")
        
        # Llamamos a la función de procesamiento. 
        # Al estar fuera del 'if button', persistirá aunque hagas clic en descargar.
        if estrategia:
          
            if "Batch" in processing_mode:
                ejecutar_batch_st(
                    df,
                    prompt_final,
                    info_llm,
                    temps,
                    message_col,
                    strategy_folder,
                    API_KEY,
                    game_col=game_col,
                    keep_cols=keep_cols,
                )
                st.session_state.proceso_finalizado = False
            else:
                st.caption("Processing row by row. To stop, click **⏹ Stop Processing** — it will halt after the current API call finishes.")
                ejecutar_procesamiento_st(
                    df,
                    prompt_final,
                    info_llm,
                    temps,
                    message_col,
                    strategy_folder,
                    API_KEY,
                    game_col=game_col,
                    keep_cols=keep_cols,
                )
        
            st.session_state.proceso_finalizado = False

        else:
            ejecutar_procesamiento_crear_st(
                df, 
                prompt_final, 
                info_llm, 
                temps, 
                message_col,  
                API_KEY
            )

        if st.button("Reset and Edit Prompt"):
            st.session_state.proceso_finalizado = False
            st.rerun()
           
def configuracion_temperaturas():
    
    # Sin crear columnas ni usar 'with'
    temps = st.multiselect("Temperatures", [0, 0.1, 0.5, 1, 1.2], default=[0])
    
        
    return temps

def ejecutar_procesamiento_st(df, prompt, config_llm, temps, message_col, strategy_folder, API_KEY, game_col=None, keep_cols=None):
    
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    proveedor = config_llm['proveedor']
    model_override = config_llm['modelo']
    keep_cols = keep_cols or []

    def enrich_row(parsed, idx, msg):
        parsed["row_id"] = idx
        parsed["original_message"] = msg
        for col in keep_cols:
            if col in df_clean.columns:
                parsed[col] = df_clean.at[idx, col]
        return parsed

    for temp in temps:
        if st.session_state.get('stop_requested'):
            break

        with st.container(border=True):
            st.subheader(f"Temperature: {temp}")

            bar = st.progress(0)
            status = st.empty()
            log_area = st.expander(f"See error logs (Temp {temp})")

            out_file = f"results_line_temp{temp}.csv"
            output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            processed_ids = set()
            rows_buffer = []

            if os.path.exists(output_path):
                try:
                    existing_df = pd.read_csv(output_path, encoding='utf-8')
                    if "row_id" in existing_df.columns:
                        processed_ids = set(existing_df["row_id"].tolist())
                        rows_buffer = existing_df.to_dict('records')
                        status.info(f"Previous progress: {len(processed_ids)} detected lines.")
                except Exception as e:
                    status.warning(f"Could not read existing file (will restart): {e}")
                    processed_ids = set()
                    rows_buffer = []

            pending_rows = [
                (idx, row[message_col], row.get(game_col, None) if game_col else None)
                for idx, row in df_clean.iterrows()
                if idx not in processed_ids
            ]

            total_total = len(df_clean)

            if not pending_rows and len(processed_ids) >= total_total:
                bar.progress(1.0)
                status.success(f"Temperature {temp} completed previously.")
            else:
                stopped_early = False

                if proveedor == "gemini":
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = {
                            executor.submit(
                                call_llm_for_message, prompt, msg, temp, "gemini", API_KEY,
                                game, model_override, "user"
                            ): (idx, msg)
                            for idx, msg, game in pending_rows
                        }
                        for i, future in enumerate(as_completed(futures)):
                            if st.session_state.get('stop_requested'):
                                executor.shutdown(wait=False, cancel_futures=True)
                                stopped_early = True
                                break
                            idx, msg = futures[future]
                            try:
                                ans = future.result()
                                parsed = parse_llm_dict(ans)
                                rows_buffer.append(enrich_row(parsed, idx, msg))
                                if len(rows_buffer) % 10 == 0:
                                    pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                                completados = len(processed_ids) + i + 1
                                bar.progress(completados / total_total)
                                status.markdown(f"**Gemini:** `{completados}/{total_total}`")
                            except Exception as e:
                                log_area.error(f"Row {idx}: {e}")

                elif proveedor == "claude":
                    for i, (idx, msg, game) in enumerate(pending_rows):
                        if st.session_state.get('stop_requested'):
                            stopped_early = True
                            break
                        try:
                            ans = call_llm_for_message(
                                prompt, msg, temp, "claude", "user", game, model_override, API_KEY, "user"
                            )
                            parsed = parse_llm_dict(ans)
                            rows_buffer.append(enrich_row(parsed, idx, msg))
                            if len(rows_buffer) % 5 == 0:
                                pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                            completados = len(processed_ids) + i + 1
                            bar.progress(completados / total_total)
                            status.markdown(f"**Claude:** `{completados}/{total_total}`")
                        except Exception as e:
                            log_area.error(f"Row {idx}: {e}")

                else:
                    for i, (idx, msg, game) in enumerate(pending_rows):
                        if st.session_state.get('stop_requested'):
                            stopped_early = True
                            break
                        try:
                            ans = call_llm_for_message(
                                prompt, msg, temp, "chatgpt", API_KEY, game, model_override, "user"
                            )
                            parsed = parse_llm_dict(ans)
                            rows_buffer.append(enrich_row(parsed, idx, msg))
                            if len(rows_buffer) % 5 == 0:
                                pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                            completados = len(processed_ids) + i + 1
                            bar.progress(completados / total_total)
                            status.markdown(f"**GPT:** `{completados}/{total_total}`")
                        except Exception as e:
                            log_area.error(f"Row {idx}: {e}")

                pd.DataFrame(rows_buffer).to_csv(output_path, index=False)

                if stopped_early:
                    status.warning(f"Stopped at row {len(rows_buffer)}. Progress saved.")
                    break

            if rows_buffer:
                pd.DataFrame(rows_buffer).to_csv(output_path, index=False)
                mostrar_boton_descarga(rows_buffer, temp, "results")

def mostrar_boton_descarga(df_temp, temp, type):
    st.session_state.proceso_finalizado = True
    
    st.success(f"✅ ¡Temp {temp} ready to be downloaded!")
    
    csv_temp = df_temp.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label=f" Download results Temp {temp}",
        data=csv_temp,
        file_name=f"{type}_temp_{temp}.csv",
        mime="text/csv",
        key=f"dl_{temp}"
    )
    
    st.divider()
           
def write_rows_to_csv(output_path, rows):
    if not rows:
        return

    file_exists = os.path.exists(output_path)
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )

def get_chatgpt_client(API_KEY):
    global llm_chatgpt
    if llm_chatgpt is None or llm_chatgpt.model_name != SELECTED_CHATGPT_MODEL:
        llm_chatgpt = ChatOpenAI(
            model=SELECTED_CHATGPT_MODEL,
            max_retries=1,
            api_key=API_KEY
        )
    return llm_chatgpt

def get_gemini_client(API_KEY):
    global llm_gemini
    if llm_gemini is None:
        llm_gemini = genai.Client(api_key=API_KEY)
    return llm_gemini

def get_claude_client(API_KEY):
    global llm_claude
    if llm_claude is None:
        if anthropic is None:
            raise ImportError("Anthropic SDK is not installed. Run 'pip install anthropic'.")
        llm_claude = anthropic.Anthropic(api_key=API_KEY)
    return llm_claude

def extract_claude_text(response):
    if hasattr(response, 'content') and response.content:
        text_blocks = [b.text for b in response.content if hasattr(b, 'type') and b.type == 'text']
        return "\n".join(text_blocks) if text_blocks else ""
    return ""

def build_classification_user_message(message, game=None):
    preamble = f"The actual game being played this period is: Game {game}\n\n" if game is not None else ""
    return (
        "Classify ONLY this message and return only a Python dictionary. "
        "Do not add explanations.\n\n" +
        preamble +
        f"Message:\n{message}"
    )

def call_llm_for_message(base_prompt, message, temp, llm, API_KEY, game= None, model_override=None, mode="user"):
    
    if game and model_override:
        user_prompt = build_classification_user_message(message, game)
    else:
        user_prompt = ""

    if llm == "gemini":
        model = model_override or SELECTED_GEMINI_MODEL
        full_prompt = base_prompt + "\n\n" + user_prompt
        response = get_gemini_client(API_KEY).models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=temp)
        )
        return response.text

    if llm == "claude":
        model = model_override or SELECTED_CLAUDE_MODEL
        response = get_claude_client(API_KEY).messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            temperature=temp,
            system=[{"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return extract_claude_text(response)

    model = model_override or SELECTED_CHATGPT_MODEL
    global llm_chatgpt
    if llm_chatgpt is None or llm_chatgpt.model_name != model:
        llm_chatgpt = get_chatgpt_client(API_KEY)

    if mode == "user":
        response = get_chatgpt_client(API_KEY).bind(temperature=temp).invoke([
            ("system", base_prompt),
            ("user", user_prompt)
        ])
    else:
        response = get_chatgpt_client(API_KEY).invoke_as_assistant(user_prompt, temperature=temp)

    return response.content

def parse_llm_dict(ans):
    try:
        start = ans.find("{")
        end   = ans.rfind("}") + 1
        raw_dict = ans[start:end]
        return ast.literal_eval(raw_dict)
    except:
        return {"error": ans[:300]}

def ejecutar_procesamiento_crear_st(df, prompt, config_llm, temps, message_col, API_KEY):    
    # Limpiamos el dataframe
    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)
    
    # Preparamos el bloque de mensajes una sola vez
    messages = df_clean[message_col].tolist()
    combined_messages = "\n".join([f"- {m}" for m in messages])
    
    full_prompt = (
        f"{prompt}\n\n"
        "These are the messages you need to analyze to create the categories:\n"
        f"{combined_messages}"
    )
    
    for temp in temps:
        with st.container(border=True):
            st.subheader(f"Results with temperature: {temp}")
            
            with st.spinner(f"The LLM is analyzing and categorizing (Temp {temp})..."):
                try:
                    # 1. Llamada única al LLM
                    proveedor = config_llm['proveedor']
                    
                    if proveedor == "gemini":
                        # Usamos tu función existente call_llm_for_message pero enviando el bloque completo
                        ans = call_llm_for_message(full_prompt, "", temp, "gemini", API_KEY)
                        
                    elif proveedor == "claude":
                        ans = call_llm_for_message(full_prompt, "", temp, "claude", API_KEY)
                    else:
                        ans = call_llm_for_message(full_prompt, "", temp, "chatgpt", API_KEY)

                    
                    # 2. Parsear el resultado (JSON -> Dict/List)
                    categorias_data = parse_llm_dict(ans)
                    
                    # 3. Mostrar de forma "bonita"
                    if categorias_data:
                        res_df = None
    
                        if isinstance(categorias_data, dict):
                            # REVISIÓN PARA TU CASO ESPECÍFICO:
                            # Convertimos el dict {Llave: Valor} en una lista de tuplas [(Llave, Valor)]
                            # y le ponemos nombres de columnas claros.
                            res_df = pd.DataFrame(
                                list(categorias_data.items()), 
                                columns=["Category", "Description"]
                            )

                        # RENDERIZADO FINAL
                        if res_df is not None:
                            # Esto imprimirá la tabla estética en vez del JSON
                            st.table(res_df) 
                            
                            # Botón de descarga usando tu función
                            mostrar_boton_descarga(res_df, temp, "categories")
                        else:
                            st.warning("No se pudo estructurar la tabla.")
                    else:
                        st.warning("The LLM did not return any data or it could not be parsed.")

                except Exception as e:
                    st.error(f"Error processing temp {temp}: {e}")    
   
# ──────────────────────────────────────────────────────────────
# BATCH HELPERS
# ──────────────────────────────────────────────────────────────
def get_openai_batch_client(API_KEY):
    global llm_openai_batch
    if llm_openai_batch is None:
        llm_openai_batch = OpenAI(api_key=API_KEY)
    return llm_openai_batch

def temp_to_id_token(temp):
    return str(temp).replace(".", "p").replace("-", "m")

def normalize_temps_for_llm(temps, llm):
    if llm != "claude":
        return temps
    valid = [t for t in temps if 0 <= float(t) <= 1]
    dropped = [t for t in temps if not (0 <= float(t) <= 1)]
    if dropped:
        st.warning(f"Claude only supports temperature 0–1. Skipping: {dropped}")
    return valid if valid else [0]

def get_openai_response_text(response_body):
    choices = response_body.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)

def get_openai_file_text(file_response):
    text_attr = getattr(file_response, "text", None)
    if callable(text_attr):
        return text_attr()
    if isinstance(text_attr, str):
        return text_attr
    if hasattr(file_response, "read"):
        data = file_response.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(file_response)

def _st_batch_status_path(provider):
    path = os.path.join(RESULTS_PATH, provider, "batch_status_st.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def load_st_batch_status(provider):
    p = _st_batch_status_path(provider)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_st_batch_status(provider, data):
    with open(_st_batch_status_path(provider), "w") as f:
        json.dump(data, f, indent=2)

def _rowmap_path(provider, batch_id):
    return os.path.join(RESULTS_PATH, provider, f"rowmap_{batch_id}.json")

# ──────────────────────────────────────────────────────────────
# BATCH SUBMISSION (Streamlit)
# ──────────────────────────────────────────────────────────────

def _to_native(v):
    if hasattr(v, "item"):
        return v.item()
    return v

def ejecutar_batch_st(df, prompt, config_llm, temps, message_col, strategy_folder, API_KEY, game_col=None, keep_cols=None):
    proveedor = config_llm['proveedor']
    model_override = config_llm['modelo']
    keep_cols = keep_cols or []
    temps = normalize_temps_for_llm(temps, proveedor)

    df_clean = df.dropna(subset=[message_col]).reset_index(drop=True)

    submitted = []

    for temp in temps:
        out_file = f"results_batch_temp{temp}.csv"
        output_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, out_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load already-processed IDs to skip them
        processed_ids = set()
        if os.path.exists(output_path):
            try:
                ex = pd.read_csv(output_path)
                if "row_id" in ex.columns:
                    processed_ids = set(int(float(v)) for v in ex["row_id"].dropna())
            except Exception:
                pass

        pending_rows = [
            (idx, row[message_col], row.get(game_col, None) if game_col else None)
            for idx, row in df_clean.iterrows()
            if idx not in processed_ids
        ]

        if not pending_rows:
            st.info(f"Temp {temp}: all rows already processed. Skipping.")
            continue

        temp_token = temp_to_id_token(temp)

        # Build row_id_map – also store keep_cols so results can be enriched later
        row_id_map = {}
        for i, (idx, msg, game) in enumerate(pending_rows):
            entry = {
                "idx": int(idx),
                "message": str(msg),
                "game": _to_native(game) if game is not None else None,
                "extra": {},
            }
            for col in keep_cols:
                if col in df_clean.columns:
                    entry["extra"][col] = _to_native(df_clean.at[idx, col])
            row_id_map[f"temp{temp_token}_row{i}"] = entry

        try:
            if proveedor == "claude":
                batch_requests = []
                for custom_id, entry in row_id_map.items():
                    user_message = build_classification_user_message(entry["message"], entry["game"])
                    batch_requests.append({
                        "custom_id": custom_id,
                        "params": {
                            "model": model_override,
                            "max_tokens": CLAUDE_MAX_TOKENS,
                            "temperature": temp,
                            "system": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
                            "messages": [{"role": "user", "content": user_message}],
                        },
                    })
                client = get_claude_client(API_KEY)
                batch_obj = client.messages.batches.create(requests=batch_requests)
                batch_id = batch_obj.id

            else:  # chatgpt
                client = get_openai_batch_client(API_KEY)
                batch_lines = []
                for custom_id, entry in row_id_map.items():
                    user_message = build_classification_user_message(entry["message"], entry["game"])
                    batch_lines.append(json.dumps({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_override,
                            "temperature": temp,
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": user_message},
                            ],
                        },
                    }, ensure_ascii=False))

                input_path = os.path.join(RESULTS_PATH, proveedor, strategy_folder, f"batch_input_temp{temp}.jsonl")
                with open(input_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(batch_lines) + "\n")

                with open(input_path, "rb") as fh:
                    input_file = client.files.create(file=fh, purpose="batch")
                batch_obj = client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                batch_id = batch_obj.id

            # Persist row_id_map to disk so it survives reruns
            with open(_rowmap_path(proveedor, batch_id), "w", encoding="utf-8") as f:
                json.dump(row_id_map, f, ensure_ascii=False, indent=2)

            # Save batch metadata
            status = load_st_batch_status(proveedor)
            status[batch_id] = {
                "batch_id": batch_id,
                "provider": proveedor,
                "model": model_override,
                "temperature": temp,
                "strategy": strategy_folder,
                "output_path": output_path,
                "total_rows": len(pending_rows),
            }
            save_st_batch_status(proveedor, status)

            submitted.append({"temp": temp, "batch_id": batch_id, "rows": len(pending_rows)})

        except Exception as e:
            st.error(f"Temp {temp}: failed to submit batch — {e}")

    if submitted:
        st.success(f"✅ {len(submitted)} batch(es) submitted successfully!")
        for s in submitted:
            st.write(f"- **Temp {s['temp']}** → `{s['batch_id']}` ({s['rows']} rows)")

        if proveedor == "claude":
            st.info(
                "**Claude Batch API** — up to **50% cheaper** than normal mode. "
                "Results ready within 24 hours.\n\n"
                "📊 Track progress at: https://console.anthropic.com/settings/workspaces/default/batches"
            )
        else:
            st.info(
                "**OpenAI Batch API** — up to **50% cheaper** than normal mode. "
                "Results ready within 24 hours.\n\n"
                "📊 Track progress at: https://platform.openai.com/batches"
            )

        st.caption("Scroll down to **Pending Batch Results** to collect results once they are ready.")
        
        mostrar_estado_batches_st(API_KEY)
        
# ──────────────────────────────────────────────────────────────
# BATCH COLLECTION (Streamlit)
# ──────────────────────────────────────────────────────────────

def mostrar_estado_batches_st(API_KEY):
    all_pending = {}
    for provider in ("claude", "chatgpt"):
        for batch_id, meta in load_st_batch_status(provider).items():
            all_pending[batch_id] = meta

    if not all_pending:
        return

    st.divider()
    st.subheader("📦 Batches Pending")
    
    # Usamos un loop más limpio
    for batch_id, meta in all_pending.items():
        with st.expander(f"🔹 {meta['provider'].upper()} - {meta['total_rows']} filas ({batch_id[:8]}...)", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            col1.caption("Configuration")
            col1.markdown(f"**Temp:** `{meta['temperature']}` | **Strategy:** `{meta['strategy']}`")
            
            col2.caption("Batch ID")
            col2.code(batch_id, language=None)
            
            if col3.button("Check & Collect", key=f"collect_{batch_id}", use_container_width=True):
                _collect_batch_results_st(batch_id, meta, API_KEY)
                                
def _collect_batch_results_st(batch_id, meta, API_KEY):
    provider = meta["provider"]
    output_path = meta["output_path"]

    rowmap_file = _rowmap_path(provider, batch_id)
    if not os.path.exists(rowmap_file):
        st.error(f"Row map file not found for batch {batch_id}. Cannot collect results.")
        return

    with open(rowmap_file, encoding="utf-8") as f:
        row_id_map = json.load(f)

    rows_buffer = []
    succeeded = 0
    errored = 0

    try:
        if provider == "claude":
            client = get_claude_client(API_KEY)
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                counts = batch.request_counts
                st.warning(
                    f"Batch not finished yet. Status: **{batch.processing_status}** | "
                    f"succeeded: {counts.succeeded} | processing: {counts.processing} | errored: {counts.errored}"
                )
                return

            for result in client.messages.batches.results(batch_id):
                if result.result.type != "succeeded":
                    errored += 1
                    continue
                entry = row_id_map.get(result.custom_id)
                if entry is None:
                    errored += 1
                    continue
                parsed = parse_llm_dict(extract_claude_text(result.result.message))
                parsed["row_id"] = entry["idx"]
                parsed["original_message"] = entry["message"]
                for col, val in entry.get("extra", {}).items():
                    parsed[col] = val
                rows_buffer.append(parsed)
                succeeded += 1

        else:  # chatgpt
            client = get_openai_batch_client(API_KEY)
            batch = client.batches.retrieve(batch_id)
            if batch.status not in ("completed", "failed", "expired", "cancelled"):
                counts = batch.request_counts
                st.warning(
                    f"Batch not finished yet. Status: **{batch.status}** | "
                    f"completed: {counts.completed} | failed: {counts.failed} | total: {counts.total}"
                )
                return

            if not batch.output_file_id:
                st.error(f"Batch ended with status `{batch.status}` and no output file.")
                _remove_st_batch(provider, batch_id)
                return

            output_text = get_openai_file_text(client.files.content(batch.output_file_id))
            for line in output_text.splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                entry = row_id_map.get(result.get("custom_id"))
                if entry is None:
                    errored += 1
                    continue
                if result.get("error") or result.get("response", {}).get("status_code") != 200:
                    errored += 1
                    continue
                ans = get_openai_response_text(result.get("response", {}).get("body", {}))
                parsed = parse_llm_dict(ans)
                parsed["row_id"] = entry["idx"]
                parsed["original_message"] = entry["message"]
                for col, val in entry.get("extra", {}).items():
                    parsed[col] = val
                rows_buffer.append(parsed)
                succeeded += 1

    except Exception as e:
        st.error(f"Error collecting batch results: {e}")
        return

# Al final, cuando ya tenemos los datos, presentamos el resultado bonito:
    if rows_buffer:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_rows_to_csv(output_path, rows_buffer)
        
        # UI limpia de éxito
        st.toast("Results collected and saved!", icon="✅")
        
        with st.container(border=True):
            
            c1, c2 = st.columns(2)
            c1.metric("Successful Rows", succeeded)
            c2.metric("Errors", errored, delta=errored, delta_color="inverse")
            
            # Previsualización opcional en un pequeño dataframe

            
            # Botón de descarga destacado
            mostrar_boton_descarga(rows_buffer, meta["temperature"], "batch_results")
            _remove_st_batch(meta["provider"], batch_id)
        
def _remove_st_batch(provider, batch_id):
    status = load_st_batch_status(provider)
    if batch_id in status:
        del status[batch_id]
        save_st_batch_status(provider, status)
    rowmap = _rowmap_path(provider, batch_id)
    if os.path.exists(rowmap):
        os.remove(rowmap)

def main():

    st.title("Responses classification with LLMs")

    # Initialize session state flags
    if 'proceso_finalizado' not in st.session_state:
        st.session_state.proceso_finalizado = False
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

    if st.button("Reset / Clear Screen"):
        folder = "Results"

        if os.path.exists(folder):
            shutil.rmtree(folder)

        os.makedirs(folder)
        st.session_state.proceso_finalizado = False
        st.session_state.stop_requested = False
        st.rerun()

    # 1. El usuario sube el archivo
    archivo_subido = st.file_uploader("Upload your CSV file", type="csv")

    if archivo_subido is not None:
        # Cargamos el dataframe para que esté disponible en las acciones
        df = pd.read_csv(archivo_subido)
        st.success("File successfully uploaded")
        
        st.write("### Preliminary data view:")
        st.dataframe(df.head())

        # 2. Preguntar qué se quiere hacer (reemplaza tus menús anteriores)
        accion = st.segmented_control(
            "What would you like to do?", 
            options=["Create categories", "Assign categories"], 
            selection_mode="single"
        )

        # 3. Lógica de ejecución según la acción seleccionada
        if accion == "Create categories":
            st.subheader("Configuration of new categories")
            #primero obtener categorias, depues hacer el mismo proceso
            menu_st(df, 1)

        elif accion == "Assign categories":
            st.subheader("Assign categories to csv")
            
            menu_st(df, 2)


    else:
        st.info("Waiting for CSV to abilitate actions")
        
if __name__ == "__main__":   
    main()
    
    
