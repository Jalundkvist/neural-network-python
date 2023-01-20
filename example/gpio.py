import RPi.GPIO as GPIO

class GPIO_:
    """ The class GPIO is a wrapper for the RPi.GPIO library.
     It allows you to control the General Purpose Input/Output (GPIO) pins on the Raspberry Pi.
    """    
    def __init__(self, pin, mode, pull_up_down=None):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.pin = pin
        GPIO.setup(self.pin, mode)
        if pull_up_down:
            GPIO.setup(self.pin, mode, pull_up_down=pull_up_down)

    def input(self):
        """input Read the value from the pin

        Returns
        -------
            1 or 0 from pin.
        """
        return GPIO.input(self.pin)

    def output(self, value):
        """output set the value of the pin

        Parameters
        ----------
        value
            1 or 0 for high or low output.
        """
        GPIO.output(self.pin, value)

    def cleanup(self):
        """cleanup the GPIO settings before exiting
        """        
        GPIO.cleanup()