
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
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the reference data
        request = refDataService.createRequest("ReferenceDataRequest")

        # Define securities
        securities = request.getElement("securities")
        securities.appendValue("GALP PL Equity")

        # Define fields
        fields = request.getElement("fields")
        
        fields.appendValue("PG_REVENUE")
        
        #fields.appendValue("SUPPLY_CHAIN_SUPPLIERS")
        #fields.appendValue("SPLC GEO")
        #fields.appendValue("SUPPLY_CHAIN_CUSTOMERS")

        # Optional - Apply overrides
        overrides = request.getElement("overrides")

        override1 = overrides.appendElement()
        override1.setElement("fieldId","PRODUCT_GEO_OVERRIDE")
        override1.setElement("value","G")

        #override2 = overrides.appendElement()
        #override2.setElement("fieldId","EQY_FUND_CRNCY")
        #override2.setElement("value","EUR")
        
        # Send the request
        print("Sending Request:", request)
        session.sendRequest(request)
        
        # Process received events
        while(True):
            ev = session.nextEvent()
            print(ev)
            print("================================")
            if ev.eventType() in [
                blpapi.Event.PARTIAL_RESPONSE,
                blpapi.Event.RESPONSE
            ]:
                for msg in ev:
                    if msg.hasElement("responseError"):
                        print(f"REQUEST FAILED: {msg.getElement('responseError')}")
                        continue
                
                    processMessage(msg)

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completley received, so we could exit
                break

    finally:
        # Stop the session
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