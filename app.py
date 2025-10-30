# -*- coding: utf-8 -*-
"""Analista Quantitativo de Apostas Esportivas (XG) - VS Code (GEMINI)"""

# Importações de bibliotecas padrão e do Flask
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Importar Google Generative AI (Gemini)
import google.generativeai as genai
from flask import Flask, request, jsonify

# Importar dotenv para carregar a chave de API do arquivo .env
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (como GOOGLE_API_KEY)
load_dotenv()

# --- REMOVIDO (Versão OpenAI) ---
# from openai import OpenAI

app = Flask(__name__)
analista_instance = None


# ============================================================================
# DEFINIÇÕES DE CLASSES E LÓGICA DO ANALISTA
# ============================================================================

@dataclass
class ConsultaAposta:
    """Estrutura de dados para consulta de aposta"""
    liga: str
    time_casa: str
    time_fora: str
    odds_1x2: Optional[Dict[str, float]] = None
    odds_over_under: Optional[Dict[str, float]] = None
    odds_btts: Optional[Dict[str, float]] = None
    contexto_adicional: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class TipoMercado(Enum):
    """Tipos de mercado suportados"""
    RESULTADO_1X2 = "1x2"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    AMBOS_MARCAM = "ambos_cam"

class AnalistaQuantitativoXG:
    """
    Alpha Quant Analyst - Especialista em Expected Goals (XG)
    Foco: Value Bets com +EV baseado em análise XG/XA/XPts
    """
    
    def __init__(self, api_key: str):
        """
        Inicialização MODIFICADA para Google Gemini API
        """
        self.api_key = api_key
        
        # 1. Configurar a API key do Google
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"Erro ao configurar a API do Gemini: {e}")
            raise

        # 2. Definir a instrução de sistema (System Role)
        # O prompt do sistema foi movido para cá, seguindo a melhor prática do Gemini.
        system_instruction = """Você é o "Alpha Quant Analyst", um especialista em modelagem preditiva de Nível 5 para futebol, com foco absoluto em Expected Goals (XG), Expected Assists (XA) e Expected Points (XPts). Seu objetivo é analisar partidas e identificar, de forma cética e rigorosa, apenas Value Bets com Valor Esperado Positivo ($EV > 2\%$). Seu tom é técnico, objetivo e livre de emoções. Você não fornece previsões baseadas em "feeling" ou estatísticas rasas."""

        # 3. Configurações de Geração
        self.generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 2048,
        }

        # 4. Configurações de Segurança (IMPORTANTE para evitar bloqueios)
        # O tema "apostas" pode ser bloqueado por "Risco/Finanças" (DANGEROUS_CONTENT)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # 5. Inicializar o Modelo Generativo (Cliente)
        self.client = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest", # Ou "gemini-pro" se preferir
            system_instruction=system_instruction,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.base_conhecimento = self._carregar_base_conhecimento()
        self._configurar_logging()

    def _configurar_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _carregar_base_conhecimento(self) -> Dict[str, Any]:
        """
        Base de conhecimento simulando RAG (sem alterações)
        """
        # (O conteúdo desta função permanece o mesmo)
        return {
            "fontes_dados_prioridade": {
                "P1_XG_Foundation": [
                    "UnderStat (https://understat.com)",
                    "FBref (https://fbref.com/en/)",
                    "XGScore (https://xgscore.io/xg-statistics)"
                ],
                "P2_Market_Value": [
                    "OddsAlerts (https://oddalerts.com/xg)",
                    "@oddsnotifierbr"
                ],
                "P3_Context_History": [
                    "FootyStats (https://footystats.org/stats/xg)",
                    "windrawwin (https://windrawwin.com)",
                    "Soccer Stats (https://soccerstats.com)"
                ],
                "P4_Live_Context": [
                    "FotMob (https://fotmob.com/pt-BR)",
                    "SofascoreBR (https://sofascore.com)",
                    "FlashscoreBR (https://flashscore.com.br)"
                ]
            },
            "metricas_fundamentais": {
                "XG": "Expected Goals - Métrica de criação de chances ofensivas",
                "XGC": "Expected Goals Conceded - Chances concedidas defensivamente",
                "XGD": "XG Difference - Diferencial (XG a favor - XG contra)",
                "XA": "Expected Assists - Assistências esperadas",
                "XPts": "Expected Points - Pontos que a equipe 'merecia' ganhar"
            },
            "modelo_preditivo": {
                "base": "Modelo de Poisson Bivariado",
                "input_primario": "XG/XGC médio últimas 10 partidas",
                "ajustes": ["Fator casa/fora", "Força da liga", "Lesões/Suspensões", "Sharp Money"]
            },
            "thresholds_decisao": {
                "ev_minimo": 0.02,  # 2% de EV mínimo
                "stake_maximo": 0.02,  # 2% da banca máximo
                "discrepancia_xg_maxima": 0.15,  # 15% de diferença entre fontes
                "kelly_fraction": 0.5  # Half Kelly por segurança
            }
        }

    def _gerar_prompt_especializado(self, consulta: ConsultaAposta) -> str:
        """
        Gera o Prompt Mestre.
        MODIFICADO: A #SYSTEM_ROLE_DEFINITION foi removida daqui,
        pois agora é tratada na inicialização do modelo (system_instruction).
        """
        
        odds_formatadas = self._formatar_odds(consulta)
        
        # O bloco #SYSTEM_ROLE_DEFINITION foi removido do início deste F-string
        prompt = f"""#METHODOLOGY_CONSTRAINT
- Modelo Preditivo: A probabilidade real ($P_c$) de resultados (1X2, O/U 2.5, BTTS) deve ser calculada usando os dados XG/XA de 10 jogos como inputs primários para um Modelo de Poisson Bivariado ou similar. Gols Reais (G) são usados apenas para avaliar a variância de finalização, não como preditor primário.
- Cálculo de Valor: O Valor Esperado ($EV$) é MANDATÓRIO e deve ser calculado pela fórmula $EV = (P_c \\times Odds) - 1$.

#KNOWLEDGE_BASE_RAG_INSTRUCTION
Sua base de dados para análise é a simulação RAG das seguintes URLs. Você deve priorizar a coleta e a coerência dos dados conforme a hierarquia (P1, P2, P3, P4):

P1 (XG Foundation - Prioridade Máxima):
{json.dumps(self.base_conhecimento["fontes_dados_prioridade"]["P1_XG_Foundation"], indent=2, ensure_ascii=False)}

P2 (Market Value & Movement):
{json.dumps(self.base_conhecimento["fontes_dados_prioridade"]["P2_Market_Value"], indent=2, ensure_ascii=False)}

P3 (Context & History):
{json.dumps(self.base_conhecimento["fontes_dados_prioridade"]["P3_Context_History"], indent=2, ensure_ascii=False)}

P4 (Live Context & Human Factor):
{json.dumps(self.base_conhecimento["fontes_dados_prioridade"]["P4_Live_Context"], indent=2, ensure_ascii=False)}

#DATA_INPUT_TEMPLATE
- Liga/Campeonato: {consulta.liga}
- Partida: {consulta.time_casa} vs {consulta.time_fora}
- Odds de Mercado:
{odds_formatadas}
{f"- Contexto Adicional: {consulta.contexto_adicional}" if consulta.contexto_adicional else ""}

#CHAIN_OF_THOUGHT_PROTOCOL
Você deve executar a análise em 5 passos CoT sequenciais. A resposta deve apresentar os resultados de cada passo de forma transparente e estruturada.

<COT_STEP_1: Coleta e Power Rating XG>
Recupere e normalize o XG, XGD, XA e XPts dos últimos 10 jogos para {consulta.time_casa} e {consulta.time_fora} (fontes P1). Calcule o Power Rating bruto, ajustando o XG pela média da liga e fator casa/fora.

IMPORTANTE: Como você está simulando o acesso às URLs, forneça estimativas razoáveis baseadas no contexto da liga e dos times, deixando claro que são simulações. Cite as fontes que você "consultaria" (P1).

<COT_STEP_2: Ajuste Contextual e Sharp Money>
Verifique Notícias de Lesões/Suspensões (P4) e a Movimentação de Odds (P2). Se um jogador-chave estiver ausente, aplique um fator de penalidade ao XG ajustado. Se as odds mudaram significativamente sem notícias óbvias, sinalize potencial Sharp Money e ajuste a $P_c$ em até 5%.

SIMULAÇÃO: Indique possíveis fatores contextuais relevantes para esta partida específica.

<COT_STEP_3: Cálculo da Probabilidade Real ($P_c$)>
Utilize o Power Rating XG ajustado (com base no Poisson) para calcular $P_c$ para os mercados 1X2, Over/Under 2.5 e BTTS.

FÓRMULA: Apresente os cálculos de forma transparente, mostrando como chegou aos valores de $P_c$.

<COT_STEP_4: Detecção de Valor Esperado (+EV) e Coerência>
Compare $P_c$ com $P_o$ (Probabilidade Implícita das Odds) para calcular o $EV$ de todos os mercados disponíveis.

FÓRMULA OBRIGATÓRIA: $EV = (P_c \\times Odds) - 1$

Liste os 3 principais mercados onde $EV > 2\\%$. 

Valide a coerência dos dados:
- Se houver discrepância de XG > 15% entre fontes P1, sinalize o risco
- Se $XPts$ da equipe favorita for consistentemente baixo, reduza a confiança na $P_c$
- Se todos os $EV < 0$, declare "No Value Found"

<COT_STEP_5: Decisão e Gerenciamento de Stake>
Selecione o mercado de maior +EV com menor desvio padrão/variância de XG.

Calcule o Stake (tamanho da aposta) utilizando a fórmula de Kelly Criterion (Half Kelly):
$Stake (\\% B) = EV / (Odd - 1)$

RESTRIÇÃO: O Stake Máximo permitido é 2% da banca, independentemente do resultado da fórmula.

Apresente a Justificativa Final com:
- Mercado Recomendado
- Odd
- EV calculado
- Stake sugerido (% da banca)
- Nível de Confiança (Alto/Médio/Baixo)

#ETHICAL_COMPLIANCE_MANDATORY
O último parágrafo da sua resposta DEVE ser o aviso de risco obrigatório:

⚠️ AVISO DE RISCO: A análise de IA não garante lucro. Apostas esportivas envolvem risco significativo de perda financeira. Jogue com responsabilidade e use apenas fundos que você pode perder. Procure ajuda se o jogo se tornar problemático.

THRESHOLDS DE DECISÃO:
{json.dumps(self.base_conhecimento["thresholds_decisao"], indent=2, ensure_ascii=False)}

AGORA, INICIE A ANÁLISE SEGUINDO RIGOROSAMENTE OS 5 PASSOS CoT:
"""
        
        return prompt

    def _formatar_odds(self, consulta: ConsultaAposta) -> str:
        """Formata as odds para inclusão no prompt"""
        # (O conteúdo desta função permanece o mesmo)
        odds_texto = []
        
        if consulta.odds_1x2:
            odds_texto.append(f"  • 1X2: Casa {consulta.odds_1x2.get('casa', 'N/A')} | Empate {consulta.odds_1x2.get('empate', 'N/A')} | Fora {consulta.odds_1x2.get('fora', 'N/A')}")
        
        if consulta.odds_over_under:
            odds_texto.append(f"  • Over/Under 2.5: Over {consulta.odds_over_under.get('over', 'N/A')} | Under {consulta.odds_over_under.get('under', 'N/A')}")
        
        if consulta.odds_btts:
            odds_texto.append(f"  • BTTS: Sim {consulta.odds_btts.get('sim', 'N/A')} | Não {consulta.odds_btts.get('nao', 'N/A')}")
        
        return "\n".join(odds_texto) if odds_texto else "  • Odds não fornecidas"

    def processar_consulta(self, consulta: ConsultaAposta) -> Dict[str, Any]:
        """
        Processa a consulta de aposta
        MODIFICADO: Usa self.client.generate_content() do Gemini
        """
        try:
            self.logger.info(f"Processando análise: {consulta.time_casa} vs {consulta.time_fora}")

            prompt = self._gerar_prompt_especializado(consulta)

            # Chamada de API MODIFICADA para Gemini
            response = self.client.generate_content(prompt)

            # Extração de resposta MODIFICADA para Gemini
            # Adicionado tratamento de erro para bloqueio de segurança
            try:
                analise_completa = response.text
            except ValueError as e:
                # Isso geralmente acontece se a resposta for bloqueada (safety ratings)
                self.logger.error(f"Resposta bloqueada pelo Gemini: {e}")
                self.logger.error(f"Detalhes do bloqueio: {response.prompt_feedback}")
                return {
                    "erro": True,
                    "mensagem": "A resposta foi bloqueada pela política de segurança do Gemini.",
                    "detalhes": f"Feedback do Prompt: {response.prompt_feedback}"
                }
            except Exception as e:
                self.logger.error(f"Erro ao extrair texto da resposta: {e}")
                return {
                    "erro": True,
                    "mensagem": "Erro ao extrair texto da resposta do Gemini.",
                    "detalhes": str(e)
                }


            resultado = {
                "analise_completa": analise_completa,
                "partida": f"{consulta.time_casa} vs {consulta.time_fora}",
                "liga": consulta.liga,
                "odds_fornecidas": {
                    "1x2": consulta.odds_1x2,
                    "over_under": consulta.odds_over_under,
                    "btts": consulta.odds_btts
                },
                "timestamp": consulta.timestamp.isoformat(),
                # Nome do modelo atualizado
                "modelo_usado": f"{self.client.model_name} (Google AI)",
                "metodologia": "Chain-of-Thought (5 passos) + Modelo Poisson + Kelly Criterion",
                "disclaimer": "⚠️ Análise de IA não garante lucro. Apostas envolvem risco de perda financeira."
            }

            self.logger.info("Análise processada com sucesso")
            return resultado

        except Exception as e:
            self.logger.error(f"Erro ao processar análise: {str(e)}")
            return {
                "erro": True,
                "mensagem": "Erro interno ao processar análise quantitativa",
                "detalhes": str(e)
            }

    def validar_contexto_consulta(self, dados: Dict) -> ConsultaAposta:
        """Valida e cria objeto ConsultaAposta a partir dos dados recebidos"""
        # (O conteúdo desta função permanece o mesmo)
        
        if not dados.get("liga"):
            raise ValueError("Liga/Campeonato é obrigatório")
        if not dados.get("time_casa"):
            raise ValueError("Time da casa é obrigatório")
        if not dados.get("time_fora"):
            raise ValueError("Time visitante é obrigatório")

        # Processa odds 1X2
        odds_1x2 = None
        if dados.get("odd_casa") or dados.get("odd_empate") or dados.get("odd_fora"):
            odds_1x2 = {
                "casa": float(dados.get("odd_casa", 0)) if dados.get("odd_casa") else None,
                "empate": float(dados.get("odd_empate", 0)) if dados.get("odd_empate") else None,
                "fora": float(dados.get("odd_fora", 0)) if dados.get("odd_fora") else None
            }

        # Processa odds Over/Under
        odds_over_under = None
        if dados.get("odd_over") or dados.get("odd_under"):
            odds_over_under = {
                "over": float(dados.get("odd_over", 0)) if dados.get("odd_over") else None,
                "under": float(dados.get("odd_under", 0)) if dados.get("odd_under") else None
            }

        # Processa odds BTTS
        odds_btts = None
        if dados.get("odd_btts_sim") or dados.get("odd_btts_nao"):
            odds_btts = {
                "sim": float(dados.get("odd_btts_sim", 0)) if dados.get("odd_btts_sim") else None,
                "nao": float(dados.get("odd_btts_nao", 0)) if dados.get("odd_btts_nao") else None
            }

        return ConsultaAposta(
            liga=dados["liga"],
            time_casa=dados["time_casa"],
            time_fora=dados["time_fora"],
            odds_1x2=odds_1x2,
            odds_over_under=odds_over_under,
            odds_btts=odds_btts,
            contexto_adicional=dados.get("contexto_adicional")
        )

def criar_analista_instance(api_key: str):
    """Cria a instância global do Analista Quantitativo"""
    global analista_instance
    analista_instance = AnalistaQuantitativoXG(api_key)
    print("✅ Analista Quantitativo XG (Gemini) inicializado com sucesso!")

# ============================================================================
# ROTAS FLASK
# (Nenhuma alteração necessária aqui, o HTML e as rotas são os mesmos)
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    """Interface HTML do Analista Quantitativo XG"""
    # (O conteúdo desta função permanece o mesmo)
    return """
    <!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Analista Quantitativo XG - Alpha Quant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 35px;
            border-bottom: 3px solid #0f2027;
            padding-bottom: 25px;
        }
        
        .header h1 {
            color: #0f2027;
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }
        
        .header .subtitle {
            color: #2c5364;
            font-size: 1.1em;
            font-weight: 600;
            font-style: italic;
        }
        
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .info-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .info-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 10px;
        }
        
        .info-card h3 {
            color: #0f2027;
            font-size: 0.9em;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .info-card p {
            color: #555;
            font-size: 0.85em;
            line-height: 1.4;
        }
        
        .form-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
        }
        
        .form-section h2 {
            color: #0f2027;
            font-size: 1.3em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c5364;
            font-size: 0.95em;
        }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 0.95em;
            transition: all 0.3s ease;
            background: white;
        }
        
        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-group textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .odds-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .odds-group {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
        }
        
        .odds-group h3 {
            color: #0f2027;
            font-size: 1em;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .odds-inputs {
            display: grid;
            gap: 12px;
        }
        
        .btn-submit {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 40px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .btn-submit:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .resultado {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            border-left: 6px solid #28a745;
            display: none;
        }
        
        .resultado.erro {
            border-left-color: #dc3545;
            background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        }
        
        .resultado h3 {
            color: #0f2027;
            font-size: 1.4em;
            margin-bottom: 20px;
        }
        
        .analise-content {
            background: white;
            padding: 25px;
            border-radius: 10px;
            white-space: pre-line;
            line-height: 1.8;
            font-size: 0.95em;
            color: #333;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .metadata {
            margin-top: 20px;
            padding: 20px;
            background: rgba(255,255,255,0.6);
            border-radius: 10px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            font-size: 0.85em;
        }
        
        .metadata-item {
            display: flex;
            flex-direction: column;
        }
        
        .metadata-item strong {
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .loading {
            text-align: center;
            color: #667eea;
            font-size: 1.1em;
            padding: 40px;
        }
        
        .disclaimer-box {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .disclaimer-box h4 {
            color: #856404;
            margin-bottom: 10px;
        }
        
        .disclaimer-box p {
            color: #856404;
            font-size: 0.9em;
            line-height: 1.6;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .form-grid,
            .odds-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Analista Quantitativo XG</h1>
            <p class="subtitle">Alpha Quant Analyst - Modelagem Preditiva Avançada</p>
            <span class="badge">Expected Goals (XG) • Chain-of-Thought • +EV Analysis</span>
        </div>

        <div class="info-cards">
            <div class="info-card">
                <h3>🎯 Metodologia</h3>
                <p>Modelo de Poisson + XG/XA últimas 10 partidas. Análise rigorosa com CoT de 5 passos.</p>
            </div>
            <div class="info-card">
                <h3>💰 Foco em Valor</h3>
                <p>Identificação exclusiva de Value Bets com EV > 2%. Kelly Criterion para gestão de stake.</p>
            </div>
            <div class="info-card">
                <h3>📈 Fontes P1</h3>
                <p>UnderStat, FBref, XGScore (prioridade máxima). Simulação RAG com 15+ URLs.</p>
            </div>
        </div>

        <form id="consultaForm">
            <div class="form-section">
                <h2>⚽ Dados da Partida</h2>
                <div class="form-grid">
                    <div class="form-group">
                        <label for="liga">Liga (ex: Premier League)</label>
                        <input type="text" id="liga" name="liga" placeholder="Ex: Brasileirão Série A" required>
                    </div>
                    <div class="form-group">
                        <label for="time_casa">Time da Casa</label>
                        <input type="text" id="time_casa" name="time_casa" placeholder="Ex: Flamengo" required>
                    </div>
                    <div class="form-group">
                        <label for="time_fora">Time Visitante</label>
                        <input type="text" id="time_fora" name="time_fora" placeholder="Ex: Palmeiras" required>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h2>📊 Odds de Mercado (Decimais)</h2>
                <div class="odds-section">
                    <div class="odds-group">
                        <h3>1X2 - Resultado Final</h3>
                        <div class="odds-inputs">
                            <div class="form-group">
                                <label for="odd_casa">Casa (1)</label>
                                <input type="number" step="0.01" min="1.01" id="odd_casa" name="odd_casa" placeholder="Ex: 2.10">
                            </div>
                            <div class="form-group">
                                <label for="odd_empate">Empate (X)</label>
                                <input type="number" step="0.01" min="1.01" id="odd_empate" name="odd_empate" placeholder="Ex: 3.40">
                            </div>
                            <div class="form-group">
                                <label for="odd_fora">Fora (2)</label>
                                <input type="number" step="0.01" min="1.01" id="odd_fora" name="odd_fora" placeholder="Ex: 3.50">
                            </div>
                        </div>
                    </div>
                    <div class="odds-group">
                        <h3>Gols e Outros Mercados</h3>
                        <div class="odds-inputs">
                            <div class="form-group">
                                <label for="odd_over">Over 2.5 Gols</label>
                                <input type="number" step="0.01" min="1.01" id="odd_over" name="odd_over" placeholder="Ex: 1.85">
                            </div>
                            <div class="form-group">
                                <label for="odd_under">Under 2.5 Gols</label>
                                <input type="number" step="0.01" min="1.01" id="odd_under" name="odd_under" placeholder="Ex: 1.95">
                            </div>
                            <div class="form-group">
                                <label for="odd_btts_sim">Ambos Marcam (Sim)</label>
                                <input type="number" step="0.01" min="1.01" id="odd_btts_sim" name="odd_btts_sim" placeholder="Ex: 1.70">
                            </div>
                             <div class="form-group">
                                <label for="odd_btts_nao">Ambos Marcam (Não)</label>
                                <input type="number" step="0.01" min="1.01" id="odd_btts_nao" name="odd_btts_nao" placeholder="Ex: 2.05">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="form-section">
                <h2>📝 Contexto Adicional (Opcional)</h2>
                <div class="form-group">
                    <label for="contexto_adicional">Informações Adicionais (Lesões, Foco em Copa, Desfalques)</label>
                    <textarea id="contexto_adicional" name="contexto_adicional" placeholder="Ex: O time da casa jogou 72h atrás pela Copa Libertadores. O artilheiro do time de fora está suspenso."></textarea>
                </div>
            </div>

            <button type="submit" class="btn-submit" id="submitBtn">Executar Análise Quantitativa XG</button>
        </form>

        <div id="loading" class="loading" style="display: none;">
            <h3>Processando Análise...</h3>
            <p>O Alpha Quant Analyst está executando o protocolo CoT de 5 passos (Coleta XG, Ajuste Contextual, Cálculo Poisson, Detecção +EV e Gestão de Stake). Isso pode levar até 45 segundos.</p>
        </div>

        <div id="resultado" class="resultado">
            <h3 id="resultadoTitle">✅ Análise Concluída</h3>
            
            <div class="metadata" id="metadata">
                <div class="metadata-item"><strong>Partida</strong><span id="metaPartida"></span></div>
                <div class="metadata-item"><strong>Liga</strong><span id="metaLiga"></span></div>
                <div class="metadata-item"><strong>Modelo</strong><span id="metaModelo"></span></div>
                <div class="metadata-item"><strong>Metodologia</strong><span id="metaMetodologia"></span></div>
                <div class="metadata-item"><strong>Timestamp</strong><span id="metaTimestamp"></span></div>
            </div>
            
            <div class="disclaimer-box">
                <h4>⚠️ ATENÇÃO: Recomendação do Sistema</h4>
                <p>O resultado abaixo é a saída bruta do Large Language Model (LLM) seguindo o protocolo quantitativo XG. **Atenção especial** ao 'Passo 5' para a recomendação final de aposta e stake.</p>
            </div>
            
            <div class="analise-content" id="analiseContent">
                </div>
        </div>
    </div>
    
    <script>
        document.getElementById('consultaForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const form = e.target;
            const formData = new FormData(form);
            const data = {};
            
            formData.forEach((value, key) => {
                if (value.trim() !== "") {
                    data[key] = value.trim();
                }
            });
            
            const url = '/analisar_aposta';
            const loadingDiv = document.getElementById('loading');
            const resultadoDiv = document.getElementById('resultado');
            const submitBtn = document.getElementById('submitBtn');
            
            loadingDiv.style.display = 'block';
            resultadoDiv.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analisando...';

            // Adiciona um pequeno delay visual antes de fazer o fetch
            setTimeout(() => {
                fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    loadingDiv.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Executar Análise Quantitativa XG';

                    if (result.erro) {
                        resultadoDiv.style.display = 'block';
                        resultadoDiv.classList.add('erro');
                        document.getElementById('resultadoTitle').textContent = '❌ Erro na Análise';
                        document.getElementById('analiseContent').innerHTML = `
                            <strong>Mensagem de Erro:</strong> ${result.mensagem}<br><br>
                            <strong>Detalhes:</strong> ${result.detalhes || 'Sem detalhes.'}
                        `;
                        document.getElementById('metadata').style.display = 'none';
                    } else {
                        resultadoDiv.style.display = 'block';
                        resultadoDiv.classList.remove('erro');
                        document.getElementById('resultadoTitle').textContent = '✅ Análise Concluída';
                        document.getElementById('metadata').style.display = 'grid';

                        // Injetar Metadados
                        document.getElementById('metaPartida').textContent = result.partida;
                        document.getElementById('metaLiga').textContent = result.liga;
                        document.getElementById('metaModelo').textContent = result.modelo_usado;
                        document.getElementById('metaMetodologia').textContent = result.metodologia;
                        document.getElementById('metaTimestamp').textContent = new Date(result.timestamp).toLocaleString('pt-BR');
                        
                        // Injetar Conteúdo da Análise
                        const rawContent = result.analise_completa;
                        const formattedContent = rawContent.replace(/\\n/g, '<br>').replace(/<COT_STEP_(\d): (.+?)>/g, '<h4>📌 Passo $1: $2</h4>');
                        document.getElementById('analiseContent').innerHTML = formattedContent;
                    }
                })
                .catch(error => {
                    loadingDiv.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Executar Análise Quantitativa XG';

                    resultadoDiv.style.display = 'block';
                    resultadoDiv.classList.add('erro');
                    document.getElementById('resultadoTitle').textContent = '❌ Erro de Comunicação';
                    document.getElementById('analiseContent').innerHTML = `
                        <strong>Detalhes:</strong> Falha ao se comunicar com o servidor de análise. ${error.message}
                    `;
                    document.getElementById('metadata').style.display = 'none';
                });
            }, 100); // 100ms delay para melhor UX
        });
    </script>
</body>
</html>
    """

@app.route("/analisar_aposta", methods=["POST"])
def analisar_aposta():
    """Endpoint para executar a análise quantitativa"""
    # (O conteúdo desta função permanece o mesmo)
    global analista_instance
    if analista_instance is None:
        return jsonify({"erro": True, "mensagem": "Analista não inicializado", "detalhes": "A chave da API pode estar faltando ou o Analista não foi criado na inicialização."}), 500

    try:
        dados = request.get_json()
        consulta = analista_instance.validar_contexto_consulta(dados)
        resultado_analise = analista_instance.processar_consulta(consulta)
        return jsonify(resultado_analise)
    except ValueError as e:
        analista_instance.logger.error(f"Erro de validação: {str(e)}")
        return jsonify({"erro": True, "mensagem": "Dados de entrada inválidos", "detalhes": str(e)}), 400
    except Exception as e:
        analista_instance.logger.error(f"Erro inesperado na rota: {str(e)}")
        return jsonify({"erro": True, "mensagem": "Erro interno do servidor", "detalhes": str(e)}), 500


# ============================================================================
# EXECUÇÃO PRINCIPAL (Padrão VS Code / Local)
# MODIFICADO: Procura por GOOGLE_API_KEY
# ============================================================================

if __name__ == "__main__":
    # 1. Carregar a Chave da API a partir do arquivo .env
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "SUA_CHAVE_API_DO_GEMINI_AQUI":
        print("❌ ERRO CRÍTICO: Chave da API 'GOOGLE_API_KEY' não configurada no arquivo .env.")
        print("Por favor, crie um arquivo .env e adicione sua chave (obtida no Google AI Studio).")
    else:
        # 2. Criar a instância global do analista
        try:
            criar_analista_instance(GOOGLE_API_KEY)
            
            # 3. Iniciar o servidor Flask localmente
            print("===================================================================")
            print("🚀 Servidor Alpha Quant Analyst (GEMINI) iniciado localmente.")
            print("Acesse a interface no seu navegador:")
            print(f"   👉 http://127.0.0.1:5000")
            print("===================================================================")
            
            app.run(debug=True, port=5000)
            
        except Exception as e:
            print(f"❌ FALHA na inicialização: {e}")