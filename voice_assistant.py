"""
Módulo de Asistente de Voz
Maneja reconocimiento de voz y síntesis de texto a voz (TTS)
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
import logging
from typing import Callable, Optional
from config import VOICE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Asistente de voz para procesar comandos y responder con TTS
    Ejecuta en un hilo separado para no bloquear la interfaz
    """
    
    def __init__(self):
        """Inicializa el asistente de voz"""
        # Configuración de reconocimiento de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Configuración de TTS
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
        # Estado del asistente
        self.is_listening = False
        self.is_active = True
        self.keyword = VOICE_CONFIG['keyword'].upper()  # Normalizar a mayúsculas
        
        # Callback para cuando se detecta la palabra clave
        self.on_keyword_detected: Optional[Callable] = None
        
        # Hilo de escucha
        self.listener_thread = None
        
        # Ajustar el micrófono al ruido ambiente
        self._calibrate_microphone()
    
    def _configure_tts(self):
        """Configura el motor de texto a voz"""
        try:
            # Configurar velocidad y volumen
            self.tts_engine.setProperty('rate', VOICE_CONFIG['tts_rate'])
            self.tts_engine.setProperty('volume', VOICE_CONFIG['tts_volume'])
            
            # Intentar configurar voz en español
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    logger.info(f"✅ Voz configurada: {voice.name}")
                    break
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo configurar voz en español: {e}")
    
    def _calibrate_microphone(self):
        """Calibra el micrófono para ajustarse al ruido ambiente"""
        logger.info("🎤 Calibrando micrófono...")
        try:
            with self.microphone as source:
                # Ajustar para ruido ambiente y aumentar sensibilidad
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.recognizer.energy_threshold = 300  # Reducir threshold para mayor sensibilidad
                self.recognizer.dynamic_energy_threshold = True
            logger.info(f"✅ Micrófono calibrado (threshold: {self.recognizer.energy_threshold})")
        except Exception as e:
            logger.error(f"❌ Error al calibrar micrófono: {e}")
    
    def speak(self, text: str):
        """
        Convierte texto a voz
        
        Args:
            text: Texto a pronunciar
        """
        try:
            logger.info(f"🔊 Diciendo: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"❌ Error en TTS: {e}")
    
    def listen_once(self) -> Optional[str]:
        """
        Escucha una vez y retorna el texto reconocido
        
        Returns:
            Texto reconocido o None si hubo error
        """
        try:
            with self.microphone as source:
                logger.info("👂 Escuchando...")
                # Aumentar timeout y phrase_time_limit para mejor detección
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=6)
                
            # Reconocer el audio
            text = self.recognizer.recognize_google(
                audio, 
                language=VOICE_CONFIG['language'],
                show_all=False
            )
            text_upper = text.upper()
            logger.info(f"✅ Reconocido: '{text}' -> Normalizado: '{text_upper}'")
            return text_upper
            
        except sr.WaitTimeoutError:
            logger.debug("⏱️ Timeout al escuchar (sin audio detectado)")
            return None
        except sr.UnknownValueError:
            logger.debug("❓ No se pudo entender el audio (habla más claro)")
            return None
        except sr.RequestError as e:
            logger.error(f"❌ Error en servicio de reconocimiento: {e}")
            logger.warning("⚠️ Verifica tu conexión a internet para Google Speech API")
            return None
        except Exception as e:
            logger.error(f"❌ Error al escuchar: {e}")
            return None
    
    def _listen_loop(self):
        """
        Loop principal de escucha (ejecuta en hilo separado)
        Escucha continuamente por la palabra clave
        """
        logger.info(f"👂 Iniciando escucha continua de palabra clave: '{self.keyword}'")
        
        # Variantes de la keyword que el reconocimiento puede detectar
        keyword_variants = [
            'LISTO', 'LISTOS', 'ESTO', 'LIST', 'LISTÓ'
        ]
        
        while self.is_active:
            if not self.is_listening:
                time.sleep(0.5)
                continue
            
            try:
                text = self.listen_once()
                
                if text:
                    # Normalizar texto
                    text_normalizado = text.upper().strip()
                    
                    # Buscar cualquier variante de la keyword
                    keyword_detected = False
                    for variant in keyword_variants:
                        if variant in text_normalizado or \
                           any(palabra == variant for palabra in text_normalizado.split()):
                            keyword_detected = True
                            logger.info(f"🎯 Palabra clave detectada! Texto: '{text}' | Variante: '{variant}'")
                            break
                    
                    if keyword_detected:
                        # Ejecutar callback si existe
                        if self.on_keyword_detected:
                            # Ejecutar en hilo separado para no bloquear
                            callback_thread = threading.Thread(
                                target=self.on_keyword_detected,
                                daemon=True
                            )
                            callback_thread.start()
                    else:
                        logger.debug(f"📝 Texto reconocido (no es keyword): '{text}'")
                    
                    # Pausar brevemente después de detectar palabra clave
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error en loop de escucha: {e}")
                time.sleep(1)
        
        logger.info("🔒 Loop de escucha finalizado")
    
    def start_listening(self):
        """Inicia la escucha continua en un hilo separado"""
        if self.listener_thread and self.listener_thread.is_alive():
            logger.warning("⚠️ Ya hay un hilo de escucha activo")
            return
        
        self.is_listening = True
        self.listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="VoiceAssistantThread"
        )
        self.listener_thread.start()
        logger.info("✅ Asistente de voz iniciado")
        
        # Mensaje de bienvenida
        self.speak("Bienvenido a Bazar Gulpery. Sistema de punto de venta activado. Di LISTO cuando termines tu compra.")
    
    def stop_listening(self):
        """Detiene la escucha continua"""
        self.is_listening = False
        logger.info("⏸️ Escucha pausada")
        
        # Mensaje de despedida
        self.speak("Hasta pronto. Gracias por usar Bazar Gulpery.")
    
    def shutdown(self):
        """Apaga completamente el asistente"""
        self.is_active = False
        self.is_listening = False
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        logger.info("🔒 Asistente de voz apagado")
    
    def announce_sale(self, productos: list, total: float):
        """
        Anuncia una venta completada
        
        Args:
            productos: Lista de nombres de productos
            total: Total de la venta
        """
        # Construir mensaje
        if not productos:
            mensaje = "No se detectaron productos en este momento."
        else:
            productos_texto = ", ".join(productos)
            mensaje = (
                f"Compra finalizada. He detectado los siguientes productos: "
                f"{productos_texto}. "
                f"El total a pagar en Bazar Gulpery es {total:.2f} dólares. "
                f"Gracias por su compra."
            )
        
        # Pronunciar mensaje
        self.speak(mensaje)
    
    def announce_status(self, message: str):
        """
        Anuncia un mensaje de estado
        
        Args:
            message: Mensaje a anunciar
        """
        self.speak(message)
    
    def shutdown(self):
        """Apaga el asistente de voz y dice adiós"""
        self.stop_listening()
        self.is_active = False
        logger.info("🔴 Asistente de voz apagado")


class VoiceCommandHandler:
    """
    Manejador de comandos de voz
    Procesa comandos específicos del sistema
    """
    
    def __init__(self, assistant: VoiceAssistant):
        """
        Inicializa el manejador de comandos
        
        Args:
            assistant: Instancia del asistente de voz
        """
        self.assistant = assistant
        self.commands = {
            'AYUDA': self._help_command,
            'ESTADO': self._status_command,
            'TOTAL': self._total_command,
        }
    
    def process_command(self, command: str) -> bool:
        """
        Procesa un comando de voz
        
        Args:
            command: Comando en texto
            
        Returns:
            True si el comando fue procesado, False en caso contrario
        """
        command = command.upper().strip()
        
        for keyword, handler in self.commands.items():
            if keyword in command:
                handler()
                return True
        
        return False
    
    def _help_command(self):
        """Comando de ayuda"""
        self.assistant.speak(
            "Comandos disponibles: di LISTO para finalizar compra, "
            "di ESTADO para ver el sistema, o di TOTAL para escuchar el total actual."
        )
    
    def _status_command(self):
        """Comando de estado del sistema"""
        self.assistant.speak("Sistema operativo. Detección de productos activa.")
    
    def _total_command(self):
        """Comando para escuchar el total"""
        # Este método será sobrescrito por la aplicación principal
        self.assistant.speak("Función de total no configurada.")
