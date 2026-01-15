      *================================================================
      * COPYBOOK: ERROR-CODES
      * Description: Standard error code definitions
      * Used by: All programs for consistent error handling
      *================================================================
       01  ERROR-CODE-DEFINITIONS.
           05  ERR-FILE-OPERATIONS.
               10  ERR-FILE-NOT-FOUND  PIC X(4) VALUE 'E001'.
               10  ERR-FILE-OPEN       PIC X(4) VALUE 'E002'.
               10  ERR-FILE-READ       PIC X(4) VALUE 'E003'.
               10  ERR-FILE-WRITE      PIC X(4) VALUE 'E004'.
               10  ERR-FILE-CLOSE      PIC X(4) VALUE 'E005'.
               10  ERR-RECORD-LOCKED   PIC X(4) VALUE 'E006'.
           05  ERR-VALIDATION.
               10  ERR-INVALID-KEY     PIC X(4) VALUE 'V001'.
               10  ERR-INVALID-DATA    PIC X(4) VALUE 'V002'.
               10  ERR-MISSING-FIELD   PIC X(4) VALUE 'V003'.
               10  ERR-RANGE-CHECK     PIC X(4) VALUE 'V004'.
               10  ERR-FORMAT-ERROR    PIC X(4) VALUE 'V005'.
           05  ERR-BUSINESS-LOGIC.
               10  ERR-CREDIT-LIMIT    PIC X(4) VALUE 'B001'.
               10  ERR-ACCOUNT-STATUS  PIC X(4) VALUE 'B002'.
               10  ERR-DUPLICATE-TRAN  PIC X(4) VALUE 'B003'.
               10  ERR-INVALID-STATE   PIC X(4) VALUE 'B004'.
           05  ERR-SYSTEM.
               10  ERR-ABEND           PIC X(4) VALUE 'S001'.
               10  ERR-TIMEOUT         PIC X(4) VALUE 'S002'.
               10  ERR-RESOURCE        PIC X(4) VALUE 'S003'.
      *
       01  ERROR-SEVERITY-CODES.
           05  SEV-INFO                PIC X(8) VALUE 'INFO    '.
           05  SEV-WARNING             PIC X(8) VALUE 'WARNING '.
           05  SEV-ERROR               PIC X(8) VALUE 'ERROR   '.
           05  SEV-CRITICAL            PIC X(8) VALUE 'CRITICAL'.
           05  SEV-FATAL               PIC X(8) VALUE 'FATAL   '.
