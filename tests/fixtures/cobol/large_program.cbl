       IDENTIFICATION DIVISION.
       PROGRAM-ID. LARGEPGM.
       AUTHOR. TEST.
       DATE-WRITTEN. 2024-01-15.
       DATE-COMPILED.
      *
      * A large COBOL program demonstrating:
      * - Complex file handling with multiple files
      * - Extensive DATA DIVISION with copybooks
      * - Multiple PROCEDURE sections
      * - Comprehensive error handling
      * - External program calls
      * - Restart/checkpoint logic
      * - Complex business logic
      *
      *===============================================================
       ENVIRONMENT DIVISION.
      *===============================================================
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-MAINFRAME.
       OBJECT-COMPUTER. IBM-MAINFRAME.
       SPECIAL-NAMES.
           DECIMAL-POINT IS COMMA.
      *
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO CUSTFILE
               ORGANIZATION IS INDEXED
               ACCESS MODE IS DYNAMIC
               RECORD KEY IS CUST-KEY
               FILE STATUS IS WS-CUST-STATUS.
           SELECT TRANSACTION-FILE
               ASSIGN TO TRANFILE
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-TRAN-STATUS.
           SELECT OUTPUT-FILE
               ASSIGN TO OUTFILE
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-OUT-STATUS.
           SELECT ERROR-FILE
               ASSIGN TO ERRFILE
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-ERR-STATUS.
           SELECT CHECKPOINT-FILE
               ASSIGN TO CHKFILE
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-CHK-STATUS.
      *
      *===============================================================
       DATA DIVISION.
      *===============================================================
       FILE SECTION.
      *
       FD  CUSTOMER-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 200 CHARACTERS.
       01  CUSTOMER-RECORD.
           05  CUST-KEY                PIC X(10).
           05  CUST-NAME               PIC X(50).
           05  CUST-ADDRESS            PIC X(80).
           05  CUST-BALANCE            PIC S9(11)V99 COMP-3.
           05  CUST-CREDIT-LIMIT       PIC S9(9)V99 COMP-3.
           05  CUST-STATUS             PIC X(2).
           05  CUST-OPEN-DATE          PIC X(10).
           05  CUST-LAST-ACTIVITY      PIC X(10).
           05  CUST-FILLER             PIC X(30).
      *
       FD  TRANSACTION-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 150 CHARACTERS.
       01  TRANSACTION-RECORD.
           05  TRAN-CUSTOMER-ID        PIC X(10).
           05  TRAN-DATE               PIC X(10).
           05  TRAN-TIME               PIC X(8).
           05  TRAN-TYPE               PIC X(2).
               88  TRAN-DEBIT          VALUE 'DR'.
               88  TRAN-CREDIT         VALUE 'CR'.
               88  TRAN-ADJUSTMENT     VALUE 'AJ'.
           05  TRAN-AMOUNT             PIC S9(9)V99 COMP-3.
           05  TRAN-REFERENCE          PIC X(20).
           05  TRAN-DESCRIPTION        PIC X(50).
           05  TRAN-STATUS             PIC X(2).
           05  TRAN-USER-ID            PIC X(8).
           05  TRAN-FILLER             PIC X(34).
      *
       FD  OUTPUT-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 200 CHARACTERS.
       01  OUTPUT-RECORD               PIC X(200).
      *
       FD  ERROR-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 250 CHARACTERS.
       01  ERROR-RECORD.
           05  ERR-TIMESTAMP           PIC X(26).
           05  ERR-SEVERITY            PIC X(8).
           05  ERR-CODE                PIC X(10).
           05  ERR-MESSAGE             PIC X(100).
           05  ERR-SOURCE-REC          PIC X(100).
           05  ERR-FILLER              PIC X(6).
      *
       FD  CHECKPOINT-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 100 CHARACTERS.
       01  CHECKPOINT-RECORD.
           05  CHK-TIMESTAMP           PIC X(26).
           05  CHK-LAST-KEY            PIC X(10).
           05  CHK-RECORDS-PROCESSED   PIC 9(10).
           05  CHK-STATUS              PIC X(2).
           05  CHK-FILLER              PIC X(52).
      *
      *---------------------------------------------------------------
       WORKING-STORAGE SECTION.
      *---------------------------------------------------------------
      *
       01  WS-FILE-STATUSES.
           05  WS-CUST-STATUS          PIC XX VALUE SPACES.
           05  WS-TRAN-STATUS          PIC XX VALUE SPACES.
           05  WS-OUT-STATUS           PIC XX VALUE SPACES.
           05  WS-ERR-STATUS           PIC XX VALUE SPACES.
           05  WS-CHK-STATUS           PIC XX VALUE SPACES.
      *
       01  WS-FLAGS.
           05  WS-EOF-TRAN-FLAG        PIC X VALUE 'N'.
               88  EOF-TRANSACTIONS    VALUE 'Y'.
               88  NOT-EOF-TRANS       VALUE 'N'.
           05  WS-ABORT-FLAG           PIC X VALUE 'N'.
               88  ABORT-PROCESSING    VALUE 'Y'.
               88  CONTINUE-PROCESSING VALUE 'N'.
           05  WS-RESTART-FLAG         PIC X VALUE 'N'.
               88  IS-RESTART          VALUE 'Y'.
               88  NOT-RESTART         VALUE 'N'.
      *
       01  WS-COUNTERS.
           05  WS-TRAN-READ            PIC 9(10) VALUE ZEROS.
           05  WS-TRAN-PROCESSED       PIC 9(10) VALUE ZEROS.
           05  WS-TRAN-ERRORS          PIC 9(10) VALUE ZEROS.
           05  WS-CUST-UPDATED         PIC 9(10) VALUE ZEROS.
           05  WS-CUST-NOT-FOUND       PIC 9(10) VALUE ZEROS.
           05  WS-CHECKPOINT-COUNTER   PIC 9(5) VALUE ZEROS.
      *
       01  WS-CHECKPOINT-INTERVAL      PIC 9(5) VALUE 1000.
      *
       01  WS-WORK-AREAS.
           05  WS-CURRENT-TIMESTAMP    PIC X(26) VALUE SPACES.
           05  WS-LAST-PROCESSED-KEY   PIC X(10) VALUE SPACES.
           05  WS-NEW-BALANCE          PIC S9(11)V99 COMP-3.
           05  WS-MESSAGE              PIC X(100) VALUE SPACES.
      *
       01  WS-VALIDATION-AREA.
           05  WS-VALID-RECORD         PIC X VALUE 'Y'.
               88  RECORD-IS-VALID     VALUE 'Y'.
               88  RECORD-IS-INVALID   VALUE 'N'.
           05  WS-ERROR-CODE           PIC X(10) VALUE SPACES.
           05  WS-ERROR-MESSAGE        PIC X(100) VALUE SPACES.
      *
       01  WS-CREDIT-CHECK.
           05  WS-OVER-LIMIT-FLAG      PIC X VALUE 'N'.
               88  OVER-CREDIT-LIMIT   VALUE 'Y'.
               88  WITHIN-LIMIT        VALUE 'N'.
           05  WS-AVAILABLE-CREDIT     PIC S9(11)V99 COMP-3.
      *
       01  WS-DATE-WORK.
           05  WS-SYS-DATE.
               10  WS-SYS-YEAR         PIC 9(4).
               10  WS-SYS-MONTH        PIC 9(2).
               10  WS-SYS-DAY          PIC 9(2).
           05  WS-FORMATTED-DATE       PIC X(10).
      *
       01  WS-RETURN-CODES.
           05  WS-RETURN-CODE          PIC S9(4) COMP VALUE ZEROS.
           05  WS-SQLCODE              PIC S9(9) COMP VALUE ZEROS.
      *
      *---------------------------------------------------------------
       LINKAGE SECTION.
      *---------------------------------------------------------------
       01  LS-PARM-DATA.
           05  LS-PARM-LENGTH          PIC S9(4) COMP.
           05  LS-PARM-VALUE           PIC X(100).
      *
      *===============================================================
       PROCEDURE DIVISION USING LS-PARM-DATA.
      *===============================================================
      *
      *---------------------------------------------------------------
       0000-MAIN-PROCEDURE.
      *---------------------------------------------------------------
      * Main control paragraph
      *---------------------------------------------------------------
           PERFORM 1000-INITIALIZE
           IF CONTINUE-PROCESSING
               PERFORM 2000-PROCESS-TRANSACTIONS
                   UNTIL EOF-TRANSACTIONS OR ABORT-PROCESSING
           END-IF
           PERFORM 8000-FINALIZE
           PERFORM 9000-TERMINATE
           GOBACK.
      *
      *===============================================================
      * INITIALIZATION SECTION
      *===============================================================
       1000-INITIALIZE.
      *---------------------------------------------------------------
      * Initialize program, open files, check for restart
      *---------------------------------------------------------------
           PERFORM 1100-GET-TIMESTAMP
           PERFORM 1200-PARSE-PARAMETERS
           PERFORM 1300-OPEN-FILES
           IF CONTINUE-PROCESSING
               PERFORM 1400-CHECK-RESTART
               PERFORM 1500-READ-FIRST-TRANSACTION
           END-IF.
      *
       1100-GET-TIMESTAMP.
           ACCEPT WS-SYS-DATE FROM DATE YYYYMMDD
           MOVE FUNCTION CURRENT-DATE TO WS-CURRENT-TIMESTAMP
           STRING WS-SYS-YEAR '-' WS-SYS-MONTH '-' WS-SYS-DAY
               DELIMITED BY SIZE
               INTO WS-FORMATTED-DATE.
      *
       1200-PARSE-PARAMETERS.
           IF LS-PARM-LENGTH > 0
               IF LS-PARM-VALUE(1:7) = 'RESTART'
                   SET IS-RESTART TO TRUE
                   DISPLAY 'RESTART MODE ENABLED'
               END-IF
           END-IF.
      *
       1300-OPEN-FILES.
           OPEN INPUT TRANSACTION-FILE
           IF WS-TRAN-STATUS NOT = '00'
               PERFORM 7100-LOG-FILE-ERROR
               SET ABORT-PROCESSING TO TRUE
           END-IF

           IF CONTINUE-PROCESSING
               OPEN I-O CUSTOMER-FILE
               IF WS-CUST-STATUS NOT = '00'
                   PERFORM 7100-LOG-FILE-ERROR
                   SET ABORT-PROCESSING TO TRUE
               END-IF
           END-IF

           IF CONTINUE-PROCESSING
               OPEN OUTPUT OUTPUT-FILE
               IF WS-OUT-STATUS NOT = '00'
                   PERFORM 7100-LOG-FILE-ERROR
                   SET ABORT-PROCESSING TO TRUE
               END-IF
           END-IF

           IF CONTINUE-PROCESSING
               OPEN OUTPUT ERROR-FILE
               IF WS-ERR-STATUS NOT = '00'
                   PERFORM 7100-LOG-FILE-ERROR
                   SET ABORT-PROCESSING TO TRUE
               END-IF
           END-IF

           IF CONTINUE-PROCESSING
               IF IS-RESTART
                   OPEN INPUT CHECKPOINT-FILE
               ELSE
                   OPEN OUTPUT CHECKPOINT-FILE
               END-IF
               IF WS-CHK-STATUS NOT = '00'
                   PERFORM 7100-LOG-FILE-ERROR
                   SET ABORT-PROCESSING TO TRUE
               END-IF
           END-IF.
      *
       1400-CHECK-RESTART.
           IF IS-RESTART
               READ CHECKPOINT-FILE INTO CHECKPOINT-RECORD
               IF WS-CHK-STATUS = '00'
                   MOVE CHK-LAST-KEY TO WS-LAST-PROCESSED-KEY
                   MOVE CHK-RECORDS-PROCESSED TO WS-TRAN-PROCESSED
                   DISPLAY 'RESTART FROM KEY: ' WS-LAST-PROCESSED-KEY
               ELSE
                   DISPLAY 'WARNING: CHECKPOINT READ FAILED'
                   SET NOT-RESTART TO TRUE
               END-IF
               CLOSE CHECKPOINT-FILE
               OPEN OUTPUT CHECKPOINT-FILE
           END-IF.
      *
       1500-READ-FIRST-TRANSACTION.
           PERFORM 2100-READ-TRANSACTION
           IF NOT-EOF-TRANS
               IF IS-RESTART
                   PERFORM 1510-SKIP-TO-RESTART-POINT
               END-IF
           END-IF.
      *
       1510-SKIP-TO-RESTART-POINT.
           PERFORM UNTIL EOF-TRANSACTIONS OR
                         TRAN-CUSTOMER-ID >= WS-LAST-PROCESSED-KEY
               PERFORM 2100-READ-TRANSACTION
           END-PERFORM.
      *
      *===============================================================
      * MAIN PROCESSING SECTION
      *===============================================================
       2000-PROCESS-TRANSACTIONS.
      *---------------------------------------------------------------
      * Process a single transaction record
      *---------------------------------------------------------------
           PERFORM 2200-VALIDATE-TRANSACTION
           IF RECORD-IS-VALID
               PERFORM 2300-LOOKUP-CUSTOMER
               IF RECORD-IS-VALID
                   PERFORM 2400-APPLY-TRANSACTION
                   IF RECORD-IS-VALID
                       PERFORM 2500-UPDATE-CUSTOMER
                       PERFORM 2600-WRITE-OUTPUT
                   END-IF
               END-IF
           END-IF
           IF RECORD-IS-INVALID
               PERFORM 7000-HANDLE-ERROR
           END-IF
           PERFORM 2700-CHECKPOINT-IF-NEEDED
           PERFORM 2100-READ-TRANSACTION.
      *
       2100-READ-TRANSACTION.
           READ TRANSACTION-FILE INTO TRANSACTION-RECORD
               AT END
                   SET EOF-TRANSACTIONS TO TRUE
               NOT AT END
                   ADD 1 TO WS-TRAN-READ
           END-READ
           IF WS-TRAN-STATUS NOT = '00' AND
              WS-TRAN-STATUS NOT = '10'
               MOVE 'ERR-READ' TO WS-ERROR-CODE
               MOVE 'Transaction file read error' TO WS-ERROR-MESSAGE
               SET ABORT-PROCESSING TO TRUE
           END-IF.
      *
       2200-VALIDATE-TRANSACTION.
           SET RECORD-IS-VALID TO TRUE
           INITIALIZE WS-ERROR-CODE WS-ERROR-MESSAGE

           IF TRAN-CUSTOMER-ID = SPACES
               SET RECORD-IS-INVALID TO TRUE
               MOVE 'VAL-001' TO WS-ERROR-CODE
               MOVE 'Customer ID is blank' TO WS-ERROR-MESSAGE
           END-IF

           IF RECORD-IS-VALID
               IF TRAN-AMOUNT NOT NUMERIC
                   SET RECORD-IS-INVALID TO TRUE
                   MOVE 'VAL-002' TO WS-ERROR-CODE
                   MOVE 'Transaction amount invalid' TO WS-ERROR-MESSAGE
               END-IF
           END-IF

           IF RECORD-IS-VALID
               IF NOT (TRAN-DEBIT OR TRAN-CREDIT OR TRAN-ADJUSTMENT)
                   SET RECORD-IS-INVALID TO TRUE
                   MOVE 'VAL-003' TO WS-ERROR-CODE
                   MOVE 'Invalid transaction type' TO WS-ERROR-MESSAGE
               END-IF
           END-IF

           IF RECORD-IS-VALID
               IF TRAN-DATE = SPACES
                   SET RECORD-IS-INVALID TO TRUE
                   MOVE 'VAL-004' TO WS-ERROR-CODE
                   MOVE 'Transaction date is blank' TO WS-ERROR-MESSAGE
               END-IF
           END-IF.
      *
       2300-LOOKUP-CUSTOMER.
           MOVE TRAN-CUSTOMER-ID TO CUST-KEY
           READ CUSTOMER-FILE
               INVALID KEY
                   SET RECORD-IS-INVALID TO TRUE
                   MOVE 'CUS-001' TO WS-ERROR-CODE
                   MOVE 'Customer not found' TO WS-ERROR-MESSAGE
                   ADD 1 TO WS-CUST-NOT-FOUND
           END-READ
           IF WS-CUST-STATUS NOT = '00' AND RECORD-IS-VALID
               SET RECORD-IS-INVALID TO TRUE
               MOVE 'CUS-002' TO WS-ERROR-CODE
               STRING 'Customer file read error: ' WS-CUST-STATUS
                   DELIMITED BY SIZE INTO WS-ERROR-MESSAGE
           END-IF.
      *
       2400-APPLY-TRANSACTION.
           EVALUATE TRUE
               WHEN TRAN-DEBIT
                   PERFORM 2410-APPLY-DEBIT
               WHEN TRAN-CREDIT
                   PERFORM 2420-APPLY-CREDIT
               WHEN TRAN-ADJUSTMENT
                   PERFORM 2430-APPLY-ADJUSTMENT
           END-EVALUATE.
      *
       2410-APPLY-DEBIT.
           COMPUTE WS-NEW-BALANCE = CUST-BALANCE - TRAN-AMOUNT
           PERFORM 2450-CHECK-CREDIT-LIMIT
           IF WITHIN-LIMIT
               MOVE WS-NEW-BALANCE TO CUST-BALANCE
           ELSE
               SET RECORD-IS-INVALID TO TRUE
               MOVE 'CRD-001' TO WS-ERROR-CODE
               MOVE 'Transaction exceeds credit limit'
                   TO WS-ERROR-MESSAGE
           END-IF.
      *
       2420-APPLY-CREDIT.
           COMPUTE CUST-BALANCE = CUST-BALANCE + TRAN-AMOUNT.
      *
       2430-APPLY-ADJUSTMENT.
           COMPUTE WS-NEW-BALANCE = CUST-BALANCE + TRAN-AMOUNT
           IF WS-NEW-BALANCE < 0
               COMPUTE WS-NEW-BALANCE = 0
           END-IF
           MOVE WS-NEW-BALANCE TO CUST-BALANCE.
      *
       2450-CHECK-CREDIT-LIMIT.
           SET WITHIN-LIMIT TO TRUE
           IF CUST-CREDIT-LIMIT > 0
               COMPUTE WS-AVAILABLE-CREDIT =
                   CUST-CREDIT-LIMIT + WS-NEW-BALANCE
               IF WS-AVAILABLE-CREDIT < 0
                   SET OVER-CREDIT-LIMIT TO TRUE
               END-IF
           END-IF.
      *
       2500-UPDATE-CUSTOMER.
           MOVE WS-FORMATTED-DATE TO CUST-LAST-ACTIVITY
           REWRITE CUSTOMER-RECORD
           IF WS-CUST-STATUS = '00'
               ADD 1 TO WS-CUST-UPDATED
               ADD 1 TO WS-TRAN-PROCESSED
           ELSE
               SET RECORD-IS-INVALID TO TRUE
               MOVE 'UPD-001' TO WS-ERROR-CODE
               STRING 'Customer update error: ' WS-CUST-STATUS
                   DELIMITED BY SIZE INTO WS-ERROR-MESSAGE
           END-IF.
      *
       2600-WRITE-OUTPUT.
           INITIALIZE OUTPUT-RECORD
           STRING TRAN-CUSTOMER-ID '|'
                  TRAN-TYPE '|'
                  TRAN-AMOUNT '|'
                  CUST-BALANCE '|'
                  WS-FORMATTED-DATE
               DELIMITED BY SIZE INTO OUTPUT-RECORD
           WRITE OUTPUT-RECORD
           IF WS-OUT-STATUS NOT = '00'
               DISPLAY 'WARNING: Output write error ' WS-OUT-STATUS
           END-IF.
      *
       2700-CHECKPOINT-IF-NEEDED.
           ADD 1 TO WS-CHECKPOINT-COUNTER
           IF WS-CHECKPOINT-COUNTER >= WS-CHECKPOINT-INTERVAL
               PERFORM 2710-WRITE-CHECKPOINT
               MOVE ZEROS TO WS-CHECKPOINT-COUNTER
           END-IF.
      *
       2710-WRITE-CHECKPOINT.
           MOVE WS-CURRENT-TIMESTAMP TO CHK-TIMESTAMP
           MOVE TRAN-CUSTOMER-ID TO CHK-LAST-KEY
           MOVE WS-TRAN-PROCESSED TO CHK-RECORDS-PROCESSED
           MOVE 'OK' TO CHK-STATUS
           WRITE CHECKPOINT-RECORD
           IF WS-CHK-STATUS NOT = '00'
               DISPLAY 'WARNING: Checkpoint write error ' WS-CHK-STATUS
           END-IF.
      *
      *===============================================================
      * ERROR HANDLING SECTION
      *===============================================================
       7000-HANDLE-ERROR.
      *---------------------------------------------------------------
      * Handle validation and processing errors
      *---------------------------------------------------------------
           ADD 1 TO WS-TRAN-ERRORS
           PERFORM 7200-WRITE-ERROR-RECORD.
      *
       7100-LOG-FILE-ERROR.
           DISPLAY 'FILE ERROR OCCURRED'
           DISPLAY '  CUST STATUS: ' WS-CUST-STATUS
           DISPLAY '  TRAN STATUS: ' WS-TRAN-STATUS
           DISPLAY '  OUT STATUS:  ' WS-OUT-STATUS
           DISPLAY '  ERR STATUS:  ' WS-ERR-STATUS
           DISPLAY '  CHK STATUS:  ' WS-CHK-STATUS.
      *
       7200-WRITE-ERROR-RECORD.
           MOVE WS-CURRENT-TIMESTAMP TO ERR-TIMESTAMP
           MOVE 'ERROR' TO ERR-SEVERITY
           MOVE WS-ERROR-CODE TO ERR-CODE
           MOVE WS-ERROR-MESSAGE TO ERR-MESSAGE
           MOVE TRANSACTION-RECORD TO ERR-SOURCE-REC
           WRITE ERROR-RECORD
           IF WS-ERR-STATUS NOT = '00'
               DISPLAY 'WARNING: Error log write failed '
                   WS-ERR-STATUS
           END-IF.
      *
      *===============================================================
      * FINALIZATION SECTION
      *===============================================================
       8000-FINALIZE.
      *---------------------------------------------------------------
      * Write final checkpoint and summary
      *---------------------------------------------------------------
           PERFORM 2710-WRITE-CHECKPOINT
           PERFORM 8100-WRITE-SUMMARY.
      *
       8100-WRITE-SUMMARY.
           DISPLAY '=========================================='
           DISPLAY 'PROCESSING SUMMARY'
           DISPLAY '=========================================='
           DISPLAY 'TRANSACTIONS READ:      ' WS-TRAN-READ
           DISPLAY 'TRANSACTIONS PROCESSED: ' WS-TRAN-PROCESSED
           DISPLAY 'TRANSACTIONS IN ERROR:  ' WS-TRAN-ERRORS
           DISPLAY 'CUSTOMERS UPDATED:      ' WS-CUST-UPDATED
           DISPLAY 'CUSTOMERS NOT FOUND:    ' WS-CUST-NOT-FOUND
           DISPLAY '=========================================='.
      *
      *===============================================================
      * TERMINATION SECTION
      *===============================================================
       9000-TERMINATE.
      *---------------------------------------------------------------
      * Close all files and set return code
      *---------------------------------------------------------------
           CLOSE TRANSACTION-FILE
           CLOSE CUSTOMER-FILE
           CLOSE OUTPUT-FILE
           CLOSE ERROR-FILE
           CLOSE CHECKPOINT-FILE

           IF ABORT-PROCESSING
               MOVE 12 TO WS-RETURN-CODE
           ELSE IF WS-TRAN-ERRORS > 0
               MOVE 4 TO WS-RETURN-CODE
           ELSE
               MOVE 0 TO WS-RETURN-CODE
           END-IF

           MOVE WS-RETURN-CODE TO RETURN-CODE.
