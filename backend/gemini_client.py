"""
Gemini Client

Generates professional insurance claim reports
using the Fusion Engine output.

Project:
Multimodal Insurance Claim Intelligence System
"""

import json
import logging
import os

from dotenv import load_dotenv
from google import genai

from backend.config import GEMINI_MODEL

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# ==========================================================
# Gemini Client
# ==========================================================

client = genai.Client(api_key=API_KEY)

# ==========================================================
# Prompt Builder
# ==========================================================


def build_prompt(fusion_result, policy_context):

    fusion_json = json.dumps(
        fusion_result,
        indent=4
    )

    prompt = f"""
You are an expert automobile insurance claim analyst.

You have two inputs.

==================================================
Fusion Engine Analysis
==================================================

{fusion_json}

==================================================
Relevant Insurance Policy Clauses
==================================================

{policy_context}

==================================================
Instructions
==================================================

1. Use ONLY the retrieved policy clauses as evidence.
2. Never invent insurance coverage.
3. Never invent exclusions.
4. If the policy clauses do not clearly determine coverage,
   explicitly mention that.
5. Explain whether the detected damages appear to be covered.
6. Mention the policy evidence whenever appropriate.
7. Return ONLY valid JSON.

Use this schema:

{{
    "executive_summary": "",
    "damage_assessment": {{
        "image_prediction": "",
        "text_prediction": "",
        "final_damages": [
            {{
                "damage": "",
                "confidence": 0.0
            }}
        ]
    }},
    "consistency_analysis": {{
        "agreement_score": 0,
        "consistency": "",
        "conflict_level": "",
        "dominant_model": "",
        "reason": ""
    }},
    "recommended_action": "",
    "next_steps": []
}}
"""

    return prompt


# ==========================================================
# Generate Report
# ==========================================================


def generate_report(fusion_result, policy_context):
    """
    Generates a structured insurance report using Gemini.

    Returns
    -------
    dict
        Structured report.
    """

    prompt = build_prompt(
    fusion_result,
    policy_context
    )

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        text = response.text.strip()

        # Remove markdown if Gemini accidentally returns it

        if text.startswith("```json"):
            text = text.replace("```json", "", 1)

        if text.startswith("```"):
            text = text.replace("```", "", 1)

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        report = json.loads(text)

        # --------------------------------------------------
        # Validation / Fallbacks
        # --------------------------------------------------

        report.setdefault("executive_summary", "")

        report.setdefault("damage_assessment", {})

        report.setdefault("consistency_analysis", {})

        report.setdefault("recommended_action", "")

        report.setdefault("next_steps", [])

        if not report["executive_summary"].strip():

            report["executive_summary"] = (
                "The insurance claim has been analyzed using image "
                "and claim text models. Please review the assessment "
                "below for details."
            )

        if not report["next_steps"]:

            report["next_steps"] = [

                "Review the submitted evidence manually.",

                "Verify the detected damages before approving the claim.",

                "Proceed according to the final assessment."
            ]

        return report

    except json.JSONDecodeError:

        logger.error("Failed to parse Gemini JSON response.")

        logger.error(text)

        return {

            "executive_summary":
            "Unable to generate AI report.",

            "damage_assessment": {

                "image_prediction":
                fusion_result["image_prediction"]["label"],

                "text_prediction":
                fusion_result["text_prediction"]["label"],

                "final_damages": [

                    {
                        "damage": damage,
                        "confidence": confidence
                    }

                    for damage, confidence
                    in fusion_result["final_damage"].items()
                ]
            },

            "consistency_analysis": {

                "agreement_score":
                fusion_result["agreement_score"],

                "consistency":
                fusion_result["consistency"],

                "conflict_level":
                fusion_result["conflict_level"],

                "dominant_model":
                fusion_result["dominant_model"],

                "reason":
                fusion_result["reason"]
            },

            "recommended_action":
            fusion_result["recommended_action"],

            "next_steps": [

                "Manual review is recommended.",

                "Verify the submitted evidence.",

                "Retry AI report generation if necessary."
            ]
        }

    except Exception as e:

        logger.exception("Gemini API Error")

        raise RuntimeError(
            f"Gemini API Error: {e}"
        ) from e


# ==========================================================
# Pretty Printer
# ==========================================================


def print_report(report):

    print("AI INSURANCE CLAIM REPORT")

    print("\nExecutive Summary")
    print(report["executive_summary"])

    print("\nDamage Assessment")

    print(
        "Image Prediction:",
        report["damage_assessment"]["image_prediction"]
    )

    print(
        "Text Prediction:",
        report["damage_assessment"]["text_prediction"]
    )

    print("\nFinal Damages")

    for damage in report["damage_assessment"]["final_damages"]:

        if isinstance(damage, dict):

            print(
                f"- {damage['damage']} "
                f"({damage['confidence']:.1%})"
            )

        else:

            print(f"- {damage}")

    print("\nConsistency Analysis")

    analysis = report["consistency_analysis"]

    print("Agreement Score :", analysis["agreement_score"])
    print("Consistency     :", analysis["consistency"])
    print("Conflict Level  :", analysis["conflict_level"])
    print("Dominant Model  :", analysis["dominant_model"])

    print("\nReason")
    print(analysis["reason"])

    print("\nRecommended Action")
    print(report["recommended_action"])

    print("\nNext Steps")

    for step in report["next_steps"]:
        print(f"• {step}")
