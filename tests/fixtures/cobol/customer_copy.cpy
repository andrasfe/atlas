      *================================================================
      * COPYBOOK: CUSTOMER-COPY
      * Description: Customer master record layout
      * Used by: Multiple programs for customer data access
      *================================================================
       01  CUSTOMER-MASTER-RECORD.
           05  CM-CUSTOMER-KEY.
               10  CM-REGION-CODE      PIC X(2).
               10  CM-CUSTOMER-NUMBER  PIC 9(8).
           05  CM-CUSTOMER-NAME.
               10  CM-LAST-NAME        PIC X(30).
               10  CM-FIRST-NAME       PIC X(20).
               10  CM-MIDDLE-INITIAL   PIC X.
           05  CM-ADDRESS.
               10  CM-STREET-LINE-1    PIC X(35).
               10  CM-STREET-LINE-2    PIC X(35).
               10  CM-CITY             PIC X(25).
               10  CM-STATE            PIC X(2).
               10  CM-ZIP-CODE         PIC X(10).
               10  CM-COUNTRY          PIC X(3).
           05  CM-CONTACT-INFO.
               10  CM-PHONE-NUMBER     PIC X(15).
               10  CM-EMAIL-ADDRESS    PIC X(50).
           05  CM-FINANCIAL-DATA.
               10  CM-BALANCE          PIC S9(11)V99 COMP-3.
               10  CM-CREDIT-LIMIT     PIC S9(9)V99 COMP-3.
               10  CM-YTD-PURCHASES    PIC S9(11)V99 COMP-3.
               10  CM-YTD-PAYMENTS     PIC S9(11)V99 COMP-3.
           05  CM-STATUS-INFO.
               10  CM-STATUS-CODE      PIC X(2).
                   88  CM-ACTIVE       VALUE 'AC'.
                   88  CM-SUSPENDED    VALUE 'SU'.
                   88  CM-CLOSED       VALUE 'CL'.
                   88  CM-PENDING      VALUE 'PE'.
               10  CM-OPEN-DATE        PIC 9(8).
               10  CM-LAST-UPDATE      PIC 9(8).
               10  CM-LAST-ACTIVITY    PIC 9(8).
           05  CM-FILLER               PIC X(50).
