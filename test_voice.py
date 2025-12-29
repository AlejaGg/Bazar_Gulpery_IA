"""
Script de Prueba de Reconocimiento de Voz
Úsalo para verificar que el micrófono y Google Speech API funcionan
"""

import torch
# Monkey-patch torch.load
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import speech_recognition as sr
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_microphone():
    """Prueba el reconocimiento de voz"""
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("="*60)
    print("PRUEBA DE RECONOCIMIENTO DE VOZ")
    print("="*60)
    print("\n🎤 Micrófonos disponibles:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")
    
    print("\n🎤 Calibrando micrófono...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
    print(f"✅ Calibrado (threshold: {recognizer.energy_threshold})")
    
    print("\n" + "="*60)
    print("INSTRUCCIONES:")
    print("  1. Habla claramente cerca del micrófono")
    print("  2. Prueba decir: CIÉRRATE")
    print("  3. Prueba decir: LISTO")
    print("  4. Presiona Ctrl+C para salir")
    print("="*60 + "\n")
    
    intentos = 0
    while True:
        try:
            intentos += 1
            print(f"\n[Intento {intentos}] 👂 Escuchando... (habla ahora)")
            
            with microphone as source:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
            
            print("🔄 Procesando audio...")
            
            # Probar reconocimiento
            text = recognizer.recognize_google(audio, language='es-ES')
            text_upper = text.upper()
            
            print(f"✅ RECONOCIDO: '{text}'")
            print(f"   Normalizado: '{text_upper}'")
            
            # Verificar keywords
            if 'CIÉRRATE' in text_upper or 'CIERRA TE' in text_upper or 'SIÉNDOTE' in text_upper:
                print("🎯 ¡Keyword CIÉRRATE detectada!")
            elif 'LISTO' in text_upper:
                print("🎯 ¡Keyword LISTO detectada!")
            else:
                print("ℹ️  No es una keyword conocida")
                
        except sr.WaitTimeoutError:
            print("⏱️ Timeout - No se detectó audio (habla más fuerte)")
        except sr.UnknownValueError:
            print("❓ No se entendió el audio (habla más claro)")
        except sr.RequestError as e:
            print(f"❌ Error de servicio: {e}")
            print("⚠️ Verifica tu conexión a internet")
            break
        except KeyboardInterrupt:
            print("\n\n👋 Prueba finalizada")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_microphone()
