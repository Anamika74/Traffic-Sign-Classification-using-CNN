import pyttsx3
import threading
import time

class TrafficVoiceAlertEngine:
    def __init__(self):
        """Initializes the offline speech synthesis parameters and actions mapping."""
        self.action_map = {
            # --- Mandatory / Regulatory Signs ---
            "STOP": "Stop sign ahead. Please apply brakes completely.",
            "GIVE_WAY": "Give way sign. Yield to traffic on the cross road.",
            "NO_ENTRY": "Entry forbidden. Do not enter this roadway.",
            "ONE_WAY": "One way traffic zone ahead.",
            "STRAIGHT_PROHIBITED": "Straight ahead route is prohibited.",
            "VEHICLES_PROHIBITED": "All motor vehicles prohibited past this point.",
            "NO_LEFT_TURN": "Left turn is prohibited ahead.",
            "NO_RIGHT_TURN": "Right turn is prohibited ahead.",
            "NO_U_TURN": "U turn is prohibited.",
            "OVERTAKING_PROHIBITED": "Overtaking prohibited. Maintain your lane position.",
            "HORN_PROHIBITED": "Silence zone. Horn prohibited.",
            "COMPULSORY_AHEAD": "Compulsory move straight ahead only.",
            "COMPULSORY_TURN_LEFT": "Compulsory turn left ahead.",
            "COMPULSORY_TURN_RIGHT": "Compulsory turn right ahead.",
            "COMPULSORY_AHEAD_OR_TURN_LEFT": "Compulsory go straight or turn left.",
            "COMPULSORY_AHEAD_OR_TURN_RIGHT": "Compulsory go straight or turn right.",
            "COMPULSORY_KEEP_LEFT": "Compulsory keep left.",
            
            # --- Speed Limits (Mandatory) ---
            "SPEED_LIMIT_20": "Speed limit twenty kilometers per hour. Slow down.",
            "SPEED_LIMIT_30": "Speed limit thirty kilometers per hour. Slow down.",
            "SPEED_LIMIT_40": "Speed limit forty kilometers per hour.",
            "SPEED_LIMIT_50": "Speed limit fifty kilometers per hour.",
            "SPEED_LIMIT_60": "Speed limit sixty kilometers per hour.",
            "SPEED_LIMIT_70": "Speed limit seventy kilometers per hour.",
            "SPEED_LIMIT_80": "Speed limit eighty kilometers per hour. High speed zone.",
            "RESTRICTION_ENDS": "All speed restrictions end here.",

            # --- Cautionary / Warning Signs ---
            "RIGHT_HAND_CURVE": "Caution, sharp right hand curve ahead.",
            "LEFT_HAND_CURVE": "Caution, sharp left hand curve ahead.",
            "HAIRPIN_BEND_RIGHT": "Warning, sharp right hairpin bend ahead.",
            "HAIRPIN_BEND_LEFT": "Warning, sharp left hairpin bend ahead.",
            "NARROW_ROAD_AHEAD": "Road narrows ahead. Watch your margins.",
            "ROAD_WIDENS_AHEAD": "Road widens ahead.",
            "NARROW_BRIDGE": "Narrow bridge ahead. Reduce speed and maintain lane position.",
            "PEDESTRIAN_CROSSING": "Pedestrian crossing ahead. Watch out for people.",
            "SCHOOL_AHEAD": "School zone ahead. Watch out for children and reduce speed.",
            "CATTLE": "Caution. Animal crossing area ahead. Drive carefully.",
            "FALLING_ROCKS": "Warning, landslide or falling rocks area ahead. Be alert.",
            "CROSS_ROAD": "Cross road intersection ahead. Scan for crossing traffic.",
            "GAP_IN_MEDIAN": "Gap in median divider ahead. Watch for turning vehicles.",
            "SIDE_ROAD_RIGHT": "Side road intersection on the right.",
            "SIDE_ROAD_LEFT": "Side road intersection on the left.",
            "STEEP_ASCENT": "Steep upward incline ahead. Prepare to climb.",
            "STEEP_DESCENT": "Steep downward slope ahead. Control your speed.",
            "SLIPPERY_ROAD": "Warning, slippery road conditions ahead. Drive smoothly.",
            "CYCLE_CROSSING": "Bicycle crossing zone ahead.",
            "MEN_AT_WORK": "Caution, construction zone ahead. Men at work.",
            "ROUNDABOUT": "Roundabout ahead. Prepare to merge or yield.",
            "DANGEROUS_DIP": "Warning, dangerous dip or speed breaker ahead.",
            "SPEED_BREAKER": "Speed breaker ahead. Reduce speed.",
            "ROUGH_ROAD": "Uneven or rough road surface ahead.",

            # --- Informatory Signs ---
            "PARKING_LOT": "Parking lot ahead.",
            "HOSPITAL": "Hospital zone ahead. Drive quietly.",
            "FIRST_AID_POST": "First aid medical post nearby.",
            "EATING_PLACE": "Restaurant or eating place ahead.",
            "PETROL_PUMP": "Fuel station detected nearby.",
        }
        
        # State tracking flags to prevent frame rate freezing and audio stutter
        self.last_spoken_alert = None
        self.is_speaking = False

    def trigger_alert(self, class_name):
        """
        Processes a model prediction. Maps it to an action sentence and fires it 
        onto an asynchronous background worker thread.
        """
        # Guard Clause: If the predicted index isn't mapped, or the speaker is busy, skip gracefully
        if class_name not in self.action_map or self.is_speaking:
            return
            
        alert_text = self.action_map[class_name]
        
        # De-duplication to prevent repetitive audio loops
        if alert_text == self.last_spoken_alert:
            return
            
        self.last_spoken_alert = alert_text
        
        # ASYNCHRONOUS THREADING: Offload speech execution so the main dashboard doesn't lag
        alert_thread = threading.Thread(target=self._speak, args=(alert_text,))
        alert_thread.daemon = True # Allows the background audio thread to close cleanly with the main app
        alert_thread.start()

    def _speak(self, text):
        """Internal synchronous worker managing the offline text-to-speech engine loop."""
        self.is_speaking = True
        try:
            # Initialize speech engine context locally inside the running worker thread
            engine = pyttsx3.init()
            
            # Calibration: Set speech speed rate (Standard natural cadence is 150-175 words per minute)
            engine.setProperty('rate', 165)
            
            print(f"🔊 System Audio Alert: '{text}'")
            engine.say(text)
            engine.runAndWait()  # Blocks ONLY the background thread while talking
        except Exception as e:
            print(f"⚠️ Voice module alert warning: {e}")
        finally:
            self.is_speaking = False

# ==============================================================================
# 🚀 LOCAL SCRIPT TEST RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("Testing ADAS Traffic Voice Alert Mapping...")
    alert_system = TrafficVoiceAlertEngine()
    
    # Simulate a pipeline identifying a STOP sign
    alert_system.trigger_alert("STOP")
    while alert_system.is_speaking:
        time.sleep(0.1) # Let it finish speaking
        
    time.sleep(0.5)
    
    # Simulate identifying a PEDESTRIAN_CROSSING right after
    alert_system.trigger_alert("PEDESTRIAN_CROSSING")
    while alert_system.is_speaking:
        time.sleep(0.1)
        
    print("System check complete.")