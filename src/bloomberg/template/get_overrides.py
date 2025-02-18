
import blpapi

def main():
    # Include example codes here
    
    sessionOptions = blpapi.SessionOptions()
    # AIK is currently optional
    # sessionOptions.setApplicationIdentityKey("<Enter AIK here>")
    
    # Create a Session
    session = blpapi.Session(sessionOptions)
    
    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/apiflds"):
            print("Failed to open //blp/apiflds")
            return

        # Obtain previously opened service
        fieldInfoService = session.getService("//blp/apiflds")

        # Create and fill the request for the reference data
        request = fieldInfoService.createRequest("FieldInfoRequest")
        
        request.append("id", "PG_REVENUE")
        # request.append("id", "EBITDA")
        # You can add more fields


        request.set("returnFieldDocumentation", True)

        print("Sending Request:", request)
        session.sendRequest(request)
        
        # Process received events
        while True:
            ev = session.nextEvent()
            
            
            for msg in ev:
                print(msg)
                
                if (msg.hasElement('fieldData')):
                    print('ahoy')

                    fieldData = msg.getElement('fieldData').getValue(0)
                    
                    
                    if (fieldData.hasElement('fieldInfo')):

                        fieldInfo = fieldData.getElement('fieldInfo')
                        
                        if (fieldInfo.hasElement('overrides')):
                            overrides = fieldInfo.getElement('overrides')
                            for ov in overrides:
                                print(ov)
                                #values = ov.getElement('validValues')
                                #print(values)

                                # IMPORTANT !!!!!!!!!!!!!!!!!!
                                # If you search the override code in Bloomberg, it says the valid Values


                if msg.hasElement("fieldInfo"):
                    fieldInfo = msg.getElement("fieldInfo")

                    print(fieldInfo)
                    
                    # Print available overrides
                    if fieldInfo.hasElement("overrides"):
                        overrides = fieldInfo.getElement("overrides")
                        print(f"Available Overrides for {field_name}:")
                        for i in range(overrides.numValues()):
                            override = overrides.getValueAsElement(i)
                            print(f"- {override.getElementAsString('mnemonic')}: {override.getElementAsString('description')}")
                    else:
                        print(f"No overrides available for {field_name}.")
            
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

    finally:
        session.stop()
                              
def processMessage(msg):
    #uncomment the line below to examine raw response from Bloomberg API
    #print(msg)
    
    securities = msg.getElement("securityData")
    numSecurities = securities.numValues()
    print(f"Processing {numSecurities} securities:")

    for i in range(numSecurities):
        security = securities.getValueAsElement(i)
        ticker = security.getElementAsString("security")
        print(f"\nTicker: {ticker}")

        if security.hasElement("securityError"):
            print(
                f"SECURITY FAILED: {security.getElement('securityError')}"
            )
            continue

        if security.hasElement("fieldData"):
            fields = security.getElement("fieldData")
            if fields.numElements() > 0:
                print("FIELD\t\tVALUE")
                print("-----\t\t-----")
                numElements = fields.numElements()
                for j in range(numElements):
                    field = fields.getElement(j)
                    print(f"{field.name()} aaa\t\t ola{field}adeus ")

        fieldExceptions = security.getElement("fieldExceptions")
        if fieldExceptions.numValues() > 0:
            print("FIELD\t\tEXCEPTION")
            print("-----\t\t---------")
            for k in range(fieldExceptions.numValues()):
                fieldException = fieldExceptions.getValueAsElement(k)
                print(
                    f"{fieldException.getElementAsString('fieldId')} "
                    f"\t\t {fieldException.getElement('errorInfo')}"
                )


main()