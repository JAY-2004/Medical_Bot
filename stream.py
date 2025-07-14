import json
import os
import pandas as pd
from ollama_embedder import CDTEmbedder
import gradio as gr
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from uuid import uuid4
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import ollama  # Import the direct ollama client

# Initialize Ollama clients
ollama_client = ollama.Client()  # For streaming
llm = Ollama(model="mistral:latest")  # For non-streaming operations
print("✅ Ollama models loaded.")

# Load CDT Embedder
cdt = CDTEmbedder("New_CDT.xlsx")

# Patient Memory Database
class PatientMemoryDB:
    def __init__(self, db_path="patient_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for patient memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP,
                last_visit TIMESTAMP
            )
        ''')
        
        # Create visits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                visit_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp TIMESTAMP,
                json_data TEXT,
                findings TEXT,
                cdt_matches TEXT,
                visit_notes TEXT
            )
        ''')
        
        # Create conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_id TEXT,
                question TEXT,
                response TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_patient(self, patient_id: str, name: str):
        """Save or update patient information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO patients (patient_id, name, created_at, last_visit)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, name, datetime.now(), datetime.now()))
        
        conn.commit()
        conn.close()
    
    def save_visit(self, visit_id: str, patient_id: str, json_data: dict, findings: list, cdt_matches: str):
        """Save a patient visit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO visits 
            (visit_id, patient_id, timestamp, json_data, findings, cdt_matches, visit_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            visit_id,
            patient_id,
            datetime.now(),
            json.dumps(json_data),
            json.dumps(findings),
            cdt_matches,
            ""
        ))
        
        # Update patient last visit
        cursor.execute('''
            UPDATE patients SET last_visit = ? WHERE patient_id = ?
        ''', (datetime.now(), patient_id))
        
        conn.commit()
        conn.close()
    
    def get_patient_history(self, patient_id: str, limit: int = 10) -> List[dict]:
        """Get patient's visit history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT visit_id, timestamp, json_data, findings, cdt_matches, visit_notes 
            FROM visits 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (patient_id, limit))
        
        visits = []
        for row in cursor.fetchall():
            visits.append({
                'visit_id': row[0],
                'timestamp': row[1],
                'json_data': json.loads(row[2]),
                'findings': json.loads(row[3]),
                'cdt_matches': row[4],
                'visit_notes': row[5]
            })
        
        conn.close()
        return visits
    
    def save_conversation(self, visit_id: str, question: str, response: str):
        """Save conversation for a visit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (visit_id, question, response, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (visit_id, question, response, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def search_patients(self, search_term: str) -> List[tuple]:
        """Search patients by name or ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT patient_id, name, last_visit 
            FROM patients 
            WHERE name LIKE ? OR patient_id LIKE ?
            ORDER BY last_visit DESC
        ''', (f'%{search_term}%', f'%{search_term}%'))
        
        results = cursor.fetchall()
        conn.close()
        return results

# Enhanced Session State with Clear Thread Types and Visit Comparison
class SessionState:
    def __init__(self):
        self.sessions = {}
        self.memory_db = PatientMemoryDB()
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'threads': {
                    'default': {
                        'json_context': None,
                        'cdt_matches': None,
                        'findings': None,
                        'chat_history': [],
                        'name': 'Default Case',
                        'json_text': "",
                        'patient_id': None,
                        'visit_id': None,
                        'thread_type': 'scenario',
                        'is_saved_visit': False
                    }
                },
                'current_thread': 'default',
                'current_patient_id': None,
                'patient_name': None
            }
        return self.sessions[session_id]
    
    def set_current_patient(self, session_id: str, patient_id: str, patient_name: str):
        """Set current patient for the session"""
        session = self.get_session(session_id)
        session['current_patient_id'] = patient_id
        session['patient_name'] = patient_name
        
        # Save patient to database
        self.memory_db.save_patient(patient_id, patient_name)
        
        # Load patient history
        history = self.memory_db.get_patient_history(patient_id)
        
        # Update current thread
        current_thread = session['threads'][session['current_thread']]
        current_thread['patient_id'] = patient_id
        current_thread['visit_id'] = str(uuid4())
        current_thread['name'] = f"📋 Visit: {patient_name} - Current Visit"
        
        return len(history)
    
    def get_patient_context(self, session_id: str) -> str:
        """Build patient context for AI from visit history"""
        session = self.get_session(session_id)
        if not session['current_patient_id']:
            return ""
        
        history = self.memory_db.get_patient_history(session['current_patient_id'])
        if not history:
            return f"New patient: {session['patient_name']} - No previous visits."
        
        context_parts = [f"Patient: {session['patient_name']} - {len(history)} previous visits"]
        
        for i, visit in enumerate(history):
            visit_date = datetime.fromisoformat(visit['timestamp']).strftime('%Y-%m-%d')
            context_parts.append(f"\n--- Visit {i+1} ({visit_date}) ---")
            
            if visit['findings']:
                context_parts.append(f"Findings: {json.dumps(visit['findings'])}")
            
            if visit['cdt_matches'] and len(visit['cdt_matches']) > 0:
                context_parts.append(f"CDT Matches: {visit['cdt_matches']}")
        
        return "\n".join(context_parts)
    
    def create_thread(self, session_id, thread_name, thread_type='scenario'):
        """Create thread with explicit type: 'scenario' or 'visit'"""
        session = self.get_session(session_id)
        thread_id = f"thread_{len(session['threads'])}"
        
        type_prefix = "🔬 Scenario: " if thread_type == 'scenario' else "📋 Visit: "
        display_name = f"{type_prefix}{thread_name}"
        
        session['threads'][thread_id] = {
            'json_context': None,
            'cdt_matches': None,
            'findings': None,
            'chat_history': [],
            'name': display_name,
            'json_text': "",
            'patient_id': session['current_patient_id'],
            'visit_id': str(uuid4()) if thread_type == 'visit' else None,
            'thread_type': thread_type,
            'is_saved_visit': False
        }
        session['current_thread'] = thread_id
        return thread_id
    
    def save_thread_as_visit(self, session_id, thread_id):
        """Convert scenario thread to actual visit and save to database"""
        session = self.get_session(session_id)
        if thread_id not in session['threads']:
            return False
            
        thread = session['threads'][thread_id]
        if not thread['json_context'] or not session['current_patient_id']:
            return False
            
        if thread['thread_type'] == 'scenario':
            thread['thread_type'] = 'visit'
            thread['visit_id'] = str(uuid4())
            thread['name'] = thread['name'].replace("🔬 Scenario: ", "📋 Visit: ")
        
        if thread['visit_id']:
            data = json.loads(thread['json_text']) if thread['json_text'] else {}
            self.memory_db.save_visit(
                thread['visit_id'],
                session['current_patient_id'],
                data,
                thread['findings'] or [],
                thread['cdt_matches'] or ""
            )
            thread['is_saved_visit'] = True
            return True
        return False
    
    def get_visit_comparison_data(self, session_id, visit_ids=None):
        """Get data for comparing multiple visits"""
        session = self.get_session(session_id)
        if not session['current_patient_id']:
            return "No patient selected for comparison."
        
        history = self.memory_db.get_patient_history(session['current_patient_id'])
        if len(history) < 2:
            return "Need at least 2 visits for comparison."
        
        if not visit_ids:
            visits_to_compare = history[:3]
        else:
            visits_to_compare = [v for v in history if v['visit_id'] in visit_ids]
        
        comparison_data = {
            'patient_name': session['patient_name'],
            'visits': []
        }
        
        for visit in visits_to_compare:
            visit_date = datetime.fromisoformat(visit['timestamp']).strftime('%Y-%m-%d %H:%M')
            visit_data = {
                'date': visit_date,
                'visit_id': visit['visit_id'],
                'findings': visit['findings'],
                'findings_count': len(visit['findings']) if visit['findings'] else 0,
                'teeth_affected': list(set([f['tooth'] for f in visit['findings']])) if visit['findings'] else [],
                'cdt_matches': visit['cdt_matches']
            }
            comparison_data['visits'].append(visit_data)
        
        return comparison_data
    
    def switch_thread(self, session_id, thread_id):
        session = self.get_session(session_id)
        if thread_id in session['threads']:
            session['current_thread'] = thread_id
    
    def get_current_thread(self, session_id):
        session = self.get_session(session_id)
        return session['threads'].get(session['current_thread'], None)
    
    def get_thread_list(self, session_id):
        session = self.get_session(session_id)
        return [{'id': tid, 'name': t['name']} for tid, t in session['threads'].items()]
    
    def clear_thread(self, session_id, thread_id):
        session = self.get_session(session_id)
        if thread_id in session['threads']:
            del session['threads'][thread_id]
            if session['current_thread'] == thread_id:
                session['current_thread'] = 'default' if 'default' in session['threads'] else next(iter(session['threads'].keys()), None)
    
    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_state = SessionState()

def save_results_to_excel(new_rows, filename="medbot_output_claude.xlsx"):
    df_new = pd.DataFrame(new_rows)

    if os.path.exists(filename):
        df_existing = pd.read_excel(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(filename, index=False)
    print(f"✅ Appended output saved to {filename}")

def extract_anomalies(data):
    """Extract ONLY anomalies that have metadata"""
    findings = []
    for tooth in data.get("teeth", []):
        number = tooth['number']
        for anom in tooth.get("anomalies", []):
            metadata = anom.get("metadata", {})
            if metadata:
                desc = anom.get("description", "")
                findings.append({
                    "tooth": number,
                    "description": desc,
                    "metadata": metadata
                })
    return findings

def format_finding_matches(findings):
    """Format ONLY findings with metadata for treatment plan"""
    formatted = []
    output_records = []
    
    if not findings:
        return "No anomalies with metadata found.", []
        
    for finding in findings:
        text = f"Tooth {finding['tooth']}: {finding['description']}"
        top_codes, _ = cdt.retrieve_best_match(text)
        
        lines = [f"Finding: {text}"]
        lines.append(f"Metadata: {json.dumps(finding['metadata'], indent=2)}")
        lines.append("Relevant CDT Codes:")
        
        if top_codes.empty:
            lines.append("No matching CDT code found.")
            cdt_codes_str = "No matching CDT code found."
        else:
            cdt_codes = []
            for _, row in top_codes.iterrows():
                lines.append(f"- {row['Code']}: {row['Description']}")
                cdt_codes.append(row['Code'])
            cdt_codes_str = ", ".join(cdt_codes)
            
        formatted.append("\n".join(lines))
        output_records.append({
            "Tooth No": finding["tooth"],
            "Anomaly Description": finding["description"],
            "Metadata": json.dumps(finding["metadata"], ensure_ascii=False),
            "CDT Codes": cdt_codes_str
        })
    
    return "\n\n".join(formatted), output_records

def json_to_full_text(data):
    """Convert entire JSON to comprehensive text context"""
    parts = []
    
    if "imagePath" in data:
        parts.append(f"Image: {data['imagePath']} ({data['imageWidth']}x{data['imageHeight']})")
    
    for tooth in data.get("teeth", []):
        number = tooth['number']
        tooth_info = [f"Tooth {number}:"]
        
        anomalies = tooth.get("anomalies", [])
        if anomalies:
            anomaly_list = []
            for anom in anomalies:
                desc = anom.get("description", "")
                metadata = anom.get("metadata", {})
                if metadata:
                    anomaly_list.append(f"{desc} (with metadata)")
                else:
                    anomaly_list.append(f"{desc}")
            tooth_info.append(f"  Anomalies: {', '.join(anomaly_list)}")
        
        procedures = tooth.get("procedures", [])
        if procedures:
            proc_list = [proc.get("description", "") for proc in procedures]
            tooth_info.append(f"  Procedures: {', '.join(proc_list)}")
        
        foreign_objects = tooth.get("foreign_objects", [])
        if foreign_objects:
            foreign_list = [obj.get("description", "") for obj in foreign_objects]
            tooth_info.append(f"  Foreign Objects: {', '.join(foreign_list)}")
        
        if len(tooth_info) > 1:
            parts.append("\n".join(tooth_info))
    
    return "\n\n".join(parts)

def enhanced_chat_with_medbot(question, chat_history, session_id):
    current_thread = session_state.get_current_thread(session_id)
    if not current_thread:
        yield "", chat_history
        return
    
    session = session_state.get_session(session_id)
    
    # Show typing indicator
    chat_history.append((question, "▌"))
    yield "", chat_history
    
    # ===== 1. CHECK IF USER EXPLICITLY REQUESTS PAST JSONS =====
    history_keywords = [
        "full history", "complete records", "raw data", "all past visits", 
        "all previous findings", "entire clinical history", "show all exams",
        "past findings", "prior records", "historical data", "old exams", 
        "earlier visits", "what was done on", "treatment history of",
        "show everything for tooth", "full records for tooth", 
        "all data on tooth", "complete history of tooth"
    ]
    is_history_request = any(keyword in question.lower() for keyword in history_keywords)
    
    if is_history_request and session['current_patient_id']:
        history = session_state.memory_db.get_patient_history(session['current_patient_id'])
        if not history:
            chat_history[-1] = (question, "No past visit data available.")
            yield "", chat_history
            return
        
        prompt = f"""
You are DentalMed AI. The user requested FULL PAST JSON DATA for analysis.

**PATIENT**: {session['patient_name']}
**TOTAL VISITS**: {len(history)}

**INSTRUCTIONS**:
1. Provide a structured overview of ALL raw JSON visits
2. Highlight key changes in anomalies/procedures
3. Do NOT summarize - show exact data differences

**FULL VISIT DATA**:
{json.dumps(history, indent=2)}

**Question**: {question}

Response:
"""
        try:
            full_response = ""
            stream = ollama_client.generate(
                model='mistral:latest',
                prompt=prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    full_response += chunk['response']
                    chat_history[-1] = (question, full_response + "▌")
                    yield "", chat_history
            
            chat_history[-1] = (question, full_response)
            yield "", chat_history
            
        except Exception as e:
            chat_history[-1] = (question, f"⚠️ Error loading full history: {str(e)}")
            yield "", chat_history
        return

    # ===== 2. VISIT COMPARISON PROMPT =====
    comparison_keywords = ["compare visits", "visit comparison", "changes since", "progression", "compare findings"]
    is_comparison_request = any(keyword in question.lower() for keyword in comparison_keywords)
    
    if is_comparison_request:
        comparison_data = session_state.get_visit_comparison_data(session_id)
        if isinstance(comparison_data, str):
            chat_history[-1] = (question, f"❌ {comparison_data}")
            yield "", chat_history
            return
        
        prompt = f"""
You are DentalMed AI. Analyze and compare multiple visits for this patient.

**PATIENT**: {comparison_data['patient_name']}
**VISITS TO COMPARE**: {len(comparison_data['visits'])} visits

**COMPARISON ANALYSIS REQUIRED**:
1. **New Findings**: What appeared in recent visits that wasn't present before?
2. **Resolved Issues**: What findings from earlier visits are no longer present?
3. **Progression**: How have existing conditions changed over time?
4. **Treatment Effectiveness**: Based on findings, how effective were previous treatments?
5. **Risk Assessment**: What patterns suggest increased/decreased risk?

**VISIT DATA**:
{json.dumps(comparison_data, indent=2)}

**FORMAT YOUR RESPONSE AS**:

## 📊 Visit Comparison Analysis

### 🆕 New Findings
- [List new findings with dates they first appeared]

### ✅ Resolved Issues  
- [List findings that are no longer present]

### 📈 Condition Progression
- [Track how conditions changed over time]

### 🎯 Treatment Effectiveness
- [Analyze treatment outcomes based on findings]

### ⚠️ Clinical Recommendations
- [Based on patterns, what should be monitored/treated]

**Your Question**: {question}

Response:
"""
        try:
            full_response = ""
            stream = ollama_client.generate(
                model='mistral:latest',
                prompt=prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    full_response += chunk['response']
                    chat_history[-1] = (question, full_response + "▌")
                    yield "", chat_history
            
            chat_history[-1] = (question, full_response)
            yield "", chat_history
            
        except Exception as e:
            chat_history[-1] = (question, f"⚠️ Error generating comparison: {str(e)}")
            yield "", chat_history
        return
    
    # Get patient context from history
    patient_context = session_state.get_patient_context(session_id)
    
    # Always use the current thread's isolated data
    json_context = current_thread.get('json_context')
    cdt_matches = current_thread.get('cdt_matches')
    findings = current_thread.get('findings', [])
    
    # ===== 3. GENERAL DENTAL QUESTIONS =====
    general_dental_keywords = ["what is", "how to", "explain", "difference between", "standard treatment"]
    is_general_question = any(keyword in question.lower() for keyword in general_dental_keywords)
    
    if is_general_question and current_thread['json_context'] is None:
        prompt = f"""
You are DentalMed AI, an expert dental assistant with comprehensive knowledge of:
- Dental procedures and terminology
- CDT codes and their applications
- Common dental conditions and treatments
- Best practices in dentistry

{patient_context if patient_context else ""}

Please provide a professional response to this question:

Question: {question}

Guidelines:
1. Be accurate and cite sources if possible
2. Use simple language for patient questions
3. Include relevant CDT codes when appropriate
4. For treatment questions, mention alternatives
5. Keep responses under 200 words unless complex

Response:
"""
        try:
            full_response = ""
            stream = ollama_client.generate(
                model='mistral:latest',
                prompt=prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    full_response += chunk['response']
                    chat_history[-1] = (question, full_response + "▌")
                    yield "", chat_history
            
            chat_history[-1] = (question, full_response)
            yield "", chat_history
            
        except Exception as e:
            chat_history[-1] = (question, f"⚠️ Error answering general question: {str(e)}")
            yield "", chat_history
        return
    
    # ===== 4. TREATMENT PLAN =====
    is_treatment_plan_request = any(keyword in question.lower() for keyword in 
                                  ["treatment plan", "create treatment", "treatment recommendation", 
                                   "cdt codes", "treatment codes"])
    
    if is_treatment_plan_request:
        current_case_context = current_thread['json_context']
        prompt = f"""
You are MedBot, a precise dental assistant AI. Create a treatment plan ONLY for teeth with anomaly metadata.

**STRICT INSTRUCTIONS**:
1. ONLY include teeth where anomalies have metadata (ignore others)
2. For each finding, provide:
   - Tooth number
   - Exact description from metadata
   - Recommended CDT code(s)
3. Format as a clean table
4. Never invent data - skip if no metadata exists
5. If no CDT code is found, write: "No matching CDT code found."
6. Provide ONLY the most clinically appropriate codes, not all possibilities.

**Teeth with Metadata**:
{current_thread['cdt_matches'] if current_thread['cdt_matches'] else "No teeth with metadata found"}

**Output Format**:
| Tooth | Finding Description | Metadata | Recommended CDT Codes |
|-------|---------------------|----------|-----------------------|
| ...   | ...                 | ...      | ...                   |

**Formatting Requirements**:
1. Keep each cell content SHORT and concise
2. Break long descriptions into key points only  
3. Each row should fit on one line

**REMEMBER**: You are making clinical decisions - choose the most appropriate treatment.

**Current Dental Case ONLY**:
{current_case_context}

**Available CDT Codes for Anomalies with Metadata**:
{current_thread['cdt_matches']}

**Your Request**: {question}

Answer:
"""
        
        if current_thread['findings']:
            _, output_records = format_finding_matches(current_thread['findings'])
            if output_records:
                save_results_to_excel(output_records)
    
    # ===== 5. DEFAULT PATIENT-SPECIFIC PROMPT =====
    else:
        prompt = f"""
You are DentalMed AI, a knowledgeable dental assistant AI with access to comprehensive patient information.

You have been provided with:
1. Complete dental case information including all teeth, anomalies, procedures, and foreign objects
2. Relevant CDT codes for anomalies that have metadata

**PATIENT HISTORY (SUMMARY)**:
{patient_context}

**CURRENT VISIT**:
{current_thread['json_context']}

**Available CDT Treatment Codes**:
{current_thread['cdt_matches']}

**Question**: {question}

Instructions:
- Use patient history to provide informed responses
- Reference previous visits when relevant
- Note any patterns or changes over time
- Provide continuity of care recommendations

**Response**:
"""
    
    try:
        full_response = ""
        stream = ollama_client.generate(
            model='mistral:latest',
            prompt=prompt,
            stream=True
        )
        
        for chunk in stream:
            if 'response' in chunk:
                full_response += chunk['response']
                chat_history[-1] = (question, full_response + "▌")
                yield "", chat_history
        
        chat_history[-1] = (question, full_response)
        current_thread['chat_history'] = chat_history
        
        # Save conversation to database
        if current_thread['visit_id']:
            session_state.memory_db.save_conversation(
                current_thread['visit_id'], 
                question, 
                full_response
            )
        
        yield "", chat_history
        
    except Exception as e:
        chat_history[-1] = (question, f"⚠️ Error processing your question: {str(e)}")
        current_thread['chat_history'] = chat_history
        yield "", chat_history

def handle_json_text_input(json_text, chat_history, session_id):
    current_thread = session_state.get_current_thread(session_id)
    session = session_state.get_session(session_id)
    
    if current_thread is None:
        return chat_history, json_text
    
    try:
        if not json_text.strip():
            raise ValueError("No JSON content provided")
            
        data = json.loads(json_text)
        
        current_thread['json_context'] = json_to_full_text(data)
        current_thread['findings'] = extract_anomalies(data)
        current_thread['json_text'] = json_text
        
        if current_thread['findings']:
            current_thread['cdt_matches'], _ = format_finding_matches(current_thread['findings'])
            context_message = f"✅ Patient data loaded in case '{current_thread['name']}'\nFound {len(current_thread['findings'])} anomalies with metadata."
        else:
            current_thread['cdt_matches'] = "No anomalies with metadata."
            context_message = f"✅ Patient data loaded in case '{current_thread['name']}'\nNo anomalies with metadata found."
        
        if session['current_patient_id'] and current_thread['visit_id']:
            session_state.memory_db.save_visit(
                current_thread['visit_id'],
                session['current_patient_id'],
                data,
                current_thread['findings'],
                current_thread['cdt_matches']
            )
            context_message += f"\n💾 Visit saved to patient history."
        
        chat_history.append(("System", context_message))
        current_thread['chat_history'] = chat_history
        
        return chat_history, json_text
        
    except json.JSONDecodeError:
        chat_history.append(("Error", "⚠️ Invalid JSON syntax."))
        return chat_history, json_text
    except Exception as e:
        chat_history.append(("Error", f"⚠️ Error: {str(e)}"))
        return chat_history, json_text

def select_patient_with_persistence(patient_input, session_id):
    """Handle patient selection with database persistence"""
    if not patient_input.strip():
        return "Please enter a patient name or ID", []
    
    results = session_state.memory_db.search_patients(patient_input)
    
    if results:
        patient_id, name, last_visit = results[0]
        visit_count = session_state.set_current_patient(session_id, patient_id, name)
        history = session_state.memory_db.get_patient_history(patient_id)
        
        message = f"✅ Selected patient: {name} (ID: {patient_id})\n📅 Last visit: {last_visit}\n📋 Total visits: {len(history)}"
        
        if history:
            history_summary = f"\n\n📋 Recent Visits:\n"
            for i, visit in enumerate(history[:3]):
                visit_date = datetime.fromisoformat(visit['timestamp']).strftime('%Y-%m-%d %H:%M')
                findings_count = len(visit['findings']) if visit['findings'] else 0
                history_summary += f"• Visit {i+1}: {visit_date} - {findings_count} findings\n"
            message += history_summary
        
        chat_history = [("System", message)]
        return message, chat_history
        
    else:
        patient_id = patient_input.strip().replace(" ", "_").lower()
        session_state.set_current_patient(session_id, patient_id, patient_input)
        
        message = f"✅ New patient created: {patient_input} (ID: {patient_id})"
        chat_history = [("System", message)]
        return message, chat_history

def get_all_patients():
    """Get all patients from database for dropdown"""
    conn = sqlite3.connect("patient_memory.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT patient_id, name, last_visit 
        FROM patients 
        ORDER BY last_visit DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    return [(f"{name} ({patient_id}) - Last: {last_visit}", patient_id) 
            for patient_id, name, last_visit in results]

def create_gradio_interface():
    with gr.Blocks(title="DentalMed AI Assistant") as demo:
        session_id = gr.State(value=lambda: str(uuid4()))
        
        gr.Markdown("# 🦷 DentalMed AI - Clinical Dental Assistant with Patient Memory")
        gr.Markdown("Select a patient, create visits or scenarios, and receive AI-powered dental recommendations with visit comparison")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                with gr.Group():
                    gr.Markdown("### 👤 Patient Selection")
                    patient_input = gr.Textbox(
                        label="Patient Name or ID",
                        placeholder="Enter patient name or ID"
                    )
                    select_patient_btn = gr.Button("Select/Create Patient", variant="primary")
                    patient_status = gr.Textbox(
                        label="Patient Status",
                        value="No patient selected",
                        interactive=False
                    )
                
                with gr.Group():
                    gr.Markdown("### 📁 Case Management")
                    
                    thread_type = gr.Radio(
                        label="Create New:",
                        choices=[
                            ("🔬 Scenario (What-if analysis)", "scenario"),
                            ("📋 Actual Visit (Save to history)", "visit")
                        ],
                        value="scenario",
                        type="value"
                    )
                    
                    new_thread_name = gr.Textbox(
                        label="Case/Visit Name",
                        placeholder="e.g., 'Root canal consultation' or 'Emergency visit'"
                    )
                    add_thread_btn = gr.Button("+ Create New", variant="secondary")
                    
                    thread_list = gr.Radio(
                        label="Active Cases",
                        choices=[],
                        value=None,
                        type="value",
                        interactive=True
                    )
                    
                    with gr.Row():
                        save_as_visit_btn = gr.Button("💾 Save as Visit", variant="primary", size="sm")
                        compare_visits_btn = gr.Button("📊 Compare Visits", variant="secondary", size="sm")
                
                json_input = gr.Textbox(
                    label="Patient Case Data (JSON Format)",
                    placeholder='Paste dental case JSON here...',
                    lines=10,
                    max_lines=20
                )
                load_json_btn = gr.Button("Load Patient Data", variant="primary")
                
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Clinical Consultation",
                    height=500,
                    show_label=True
                )
                
                msg = gr.Textbox(
                    label="Consult with DentalMed AI",
                    placeholder="Try: 'Generate treatment plan', 'Compare visits', or 'What changed since last visit?'",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Submit Query", variant="primary")
                    clear_btn = gr.Button("Clear Current Case")
                    clear_session_btn = gr.Button("New Session", variant="stop")
        
        def update_thread_list(session_id):
            threads = session_state.get_thread_list(session_id)
            choices = [t['name'] for t in threads]
            selected = threads[0]['name'] if threads else None
            return gr.update(choices=choices, value=selected)
        
        def create_new_thread(thread_name, thread_type_val, session_id, current_thread):
            if not thread_name.strip():
                raise gr.Error("Please enter a case name")
            thread_id = session_state.create_thread(session_id, thread_name, thread_type_val)
            return "", update_thread_list(session_id), []
        
        def save_current_as_visit(session_id):
            session = session_state.get_session(session_id)
            current_thread_id = session['current_thread']
            success = session_state.save_thread_as_visit(session_id, current_thread_id)
            
            if success:
                message = "✅ Scenario saved as actual visit to patient history!"
                return update_thread_list(session_id), [(None, message)]
            else:
                return gr.update(), [(None, "❌ Cannot save: No patient data or patient not selected")]
        
        def trigger_visit_comparison(session_id):
            comparison_response = enhanced_chat_with_medbot(
                "Compare visits and show progression analysis", 
                [], 
                session_id
            )
            return comparison_response[1]
        
        demo.load(
            lambda sid: update_thread_list(sid),
            inputs=[session_id],
            outputs=[thread_list]
        )
        
        select_patient_btn.click(
            select_patient_with_persistence,
            inputs=[patient_input, session_id],
            outputs=[patient_status, chatbot]
        )
        
        load_json_btn.click(
            handle_json_text_input,
            inputs=[json_input, chatbot, session_id],
            outputs=[chatbot, json_input]
        )
        
        send_btn.click(
            enhanced_chat_with_medbot,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot],
            show_progress=False
        )
        
        msg.submit(
            enhanced_chat_with_medbot,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot],
            show_progress=False
        )
        
        add_thread_btn.click(
            create_new_thread,
            inputs=[new_thread_name, thread_type, session_id, thread_list],
            outputs=[new_thread_name, thread_list, chatbot]
        )
        
        thread_list.change(
            lambda selected_thread, sid: switch_active_thread(selected_thread, sid),
            inputs=[thread_list, session_id],
            outputs=[chatbot, json_input]
        )
        
        save_as_visit_btn.click(
            save_current_as_visit,
            inputs=[session_id],
            outputs=[thread_list, chatbot]
        )
        
        compare_visits_btn.click(
            trigger_visit_comparison,
            inputs=[session_id],
            outputs=[chatbot]
        )
        
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        ).then(
            lambda sid: session_state.get_current_thread(sid)['chat_history'].clear(),
            inputs=[session_id],
            outputs=[]
        )
        
        clear_session_btn.click(
            lambda sid: session_state.clear_session(sid),
            inputs=[session_id],
            outputs=[]
        ).then(
            lambda: ([], "Session reset. Please select a patient and load case data.", "", [], None, "No patient selected"),
            outputs=[chatbot, json_input, msg, thread_list, new_thread_name, patient_status]
        )
        
        def switch_active_thread(selected_thread, session_id):
            session = session_state.get_session(session_id)
            for thread_id, thread in session['threads'].items():
                if thread['name'] == selected_thread:
                    session_state.switch_thread(session_id, thread_id)
                    current_thread = session_state.get_current_thread(session_id)
                    return current_thread['chat_history'], current_thread.get('json_text', "")
            return [], ""
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface() 
    demo.launch(share=True)