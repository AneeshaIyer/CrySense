# utils.py
RECOMMENDATIONS = {
    "hungry": "🍼 Offer feeding. Watch for rooting/sucking cues.",
    "pain": "⚠️ Check for fever, rash, swelling. Consult pediatrician if persistent.",
    "sleepy": "😴 Dim lights, swaddle, reduce stimulation.",
    "discomfort": "👕 Check diaper, clothing, room temperature.",
    "diaper": "👶 Change diaper and ensure comfort."
}

def interpret(prediction: str, confidence: float):
    urgency = "HIGH" if confidence >= 0.80 else ("MEDIUM" if confidence >= 0.60 else "LOW")
    rec = RECOMMENDATIONS.get(prediction, "No specific recommendation.")
    md = f"""### 🍼 CrySense Analysis
**Detected Category:** **{prediction}**  
**Confidence:** {confidence*100:.1f}%  
**Urgency:** {urgency}

**Recommendation:**  
{rec}
"""
    return md
