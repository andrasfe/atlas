       IDENTIFICATION DIVISION.
       PROGRAM-ID. MEDPGM.
       AUTHOR. TEST.
       DATE-WRITTEN. 2024-01-15.
      *
      * A medium-sized COBOL program demonstrating:
      * - File handling with status checks
      * - Multiple sections in DATA DIVISION
      * - Error handling patterns
      * - External program calls
      *
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-PC.
       OBJECT-COMPUTER. IBM-PC.
      *
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT INPUT-FILE
               ASSIGN TO 'INFILE'
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-INPUT-STATUS.
           SELECT OUTPUT-FILE
               ASSIGN TO 'OUTFILE'
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-OUTPUT-STATUS.
      *
       DATA DIVISION.
       FILE SECTION.
       FD  INPUT-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 80 CHARACTERS.
       01  INPUT-RECORD.
           05  IN-CUSTOMER-ID      PIC X(10).
           05  IN-CUSTOMER-NAME    PIC X(30).
           05  IN-BALANCE          PIC 9(7)V99.
           05  IN-FILLER           PIC X(31).
      *
       FD  OUTPUT-FILE
           LABEL RECORDS ARE STANDARD
           RECORD CONTAINS 120 CHARACTERS.
       01  OUTPUT-RECORD.
           05  OUT-CUSTOMER-ID     PIC X(10).
           05  OUT-CUSTOMER-NAME   PIC X(30).
           05  OUT-BALANCE         PIC Z,ZZZ,ZZ9.99.
           05  OUT-STATUS          PIC X(10).
           05  OUT-PROCESS-DATE    PIC X(10).
           05  OUT-FILLER          PIC X(47).
      *
       WORKING-STORAGE SECTION.
       01  WS-FILE-STATUSES.
           05  WS-INPUT-STATUS     PIC XX VALUE SPACES.
           05  WS-OUTPUT-STATUS    PIC XX VALUE SPACES.
      *
       01  WS-FLAGS.
           05  WS-EOF-FLAG         PIC X VALUE 'N'.
               88  EOF-REACHED     VALUE 'Y'.
               88  NOT-EOF         VALUE 'N'.
           05  WS-ERROR-FLAG       PIC X VALUE 'N'.
               88  ERROR-OCCURRED  VALUE 'Y'.
               88  NO-ERROR        VALUE 'N'.
      *
       01  WS-COUNTERS.
           05  WS-RECORDS-READ     PIC 9(7) VALUE ZEROS.
           05  WS-RECORDS-WRITTEN  PIC 9(7) VALUE ZEROS.
           05  WS-RECORDS-ERROR    PIC 9(7) VALUE ZEROS.
      *
       01  WS-WORK-AREAS.
           05  WS-CURRENT-DATE     PIC X(10) VALUE SPACES.
           05  WS-PROCESS-MESSAGE  PIC X(80) VALUE SPACES.
      *
       01  WS-VALIDATION-RESULT.
           05  WS-VALID-FLAG       PIC X VALUE 'Y'.
               88  RECORD-VALID    VALUE 'Y'.
               88  RECORD-INVALID  VALUE 'N'.
           05  WS-ERROR-CODE       PIC X(4) VALUE SPACES.
      *
       LINKAGE SECTION.
       01  LS-RETURN-CODE          PIC S9(4) COMP.
      *
       PROCEDURE DIVISION.
       0000-MAIN-PROCEDURE.
           PERFORM 1000-INITIALIZE
           PERFORM 2000-PROCESS-RECORDS
               UNTIL EOF-REACHED OR ERROR-OCCURRED
           PERFORM 8000-WRITE-SUMMARY
           PERFORM 9000-TERMINATE
           STOP RUN.
      *
       1000-INITIALIZE.
           PERFORM 1100-OPEN-FILES
           PERFORM 1200-GET-DATE
           IF NO-ERROR
               PERFORM 1300-READ-FIRST-RECORD
           END-IF.
      *
       1100-OPEN-FILES.
           OPEN INPUT INPUT-FILE
           IF WS-INPUT-STATUS NOT = '00'
               DISPLAY 'ERROR OPENING INPUT FILE: ' WS-INPUT-STATUS
               SET ERROR-OCCURRED TO TRUE
           END-IF
           IF NO-ERROR
               OPEN OUTPUT OUTPUT-FILE
               IF WS-OUTPUT-STATUS NOT = '00'
                   DISPLAY 'ERROR OPENING OUTPUT FILE: ' WS-OUTPUT-STATUS
                   SET ERROR-OCCURRED TO TRUE
               END-IF
           END-IF.
      *
       1200-GET-DATE.
           ACCEPT WS-CURRENT-DATE FROM DATE YYYYMMDD
           INSPECT WS-CURRENT-DATE REPLACING ALL '/' BY '-'.
      *
       1300-READ-FIRST-RECORD.
           PERFORM 2100-READ-INPUT
           IF NOT-EOF
               ADD 1 TO WS-RECORDS-READ
           END-IF.
      *
       2000-PROCESS-RECORDS.
           PERFORM 2200-VALIDATE-RECORD
           IF RECORD-VALID
               PERFORM 2300-TRANSFORM-RECORD
               PERFORM 2400-WRITE-OUTPUT
           ELSE
               PERFORM 2500-HANDLE-INVALID
           END-IF
           PERFORM 2100-READ-INPUT
           IF NOT-EOF
               ADD 1 TO WS-RECORDS-READ
           END-IF.
      *
       2100-READ-INPUT.
           READ INPUT-FILE
               AT END
                   SET EOF-REACHED TO TRUE
               NOT AT END
                   CONTINUE
           END-READ
           IF WS-INPUT-STATUS NOT = '00' AND
              WS-INPUT-STATUS NOT = '10'
               DISPLAY 'READ ERROR: ' WS-INPUT-STATUS
               SET ERROR-OCCURRED TO TRUE
           END-IF.
      *
       2200-VALIDATE-RECORD.
           SET RECORD-VALID TO TRUE
           MOVE SPACES TO WS-ERROR-CODE

           IF IN-CUSTOMER-ID = SPACES
               SET RECORD-INVALID TO TRUE
               MOVE 'E001' TO WS-ERROR-CODE
           END-IF

           IF RECORD-VALID
               IF IN-BALANCE NOT NUMERIC
                   SET RECORD-INVALID TO TRUE
                   MOVE 'E002' TO WS-ERROR-CODE
               END-IF
           END-IF.
      *
       2300-TRANSFORM-RECORD.
           MOVE IN-CUSTOMER-ID TO OUT-CUSTOMER-ID
           MOVE IN-CUSTOMER-NAME TO OUT-CUSTOMER-NAME
           MOVE IN-BALANCE TO OUT-BALANCE
           MOVE 'PROCESSED' TO OUT-STATUS
           MOVE WS-CURRENT-DATE TO OUT-PROCESS-DATE
           MOVE SPACES TO OUT-FILLER.
      *
       2400-WRITE-OUTPUT.
           WRITE OUTPUT-RECORD
           IF WS-OUTPUT-STATUS = '00'
               ADD 1 TO WS-RECORDS-WRITTEN
           ELSE
               DISPLAY 'WRITE ERROR: ' WS-OUTPUT-STATUS
               ADD 1 TO WS-RECORDS-ERROR
           END-IF.
      *
       2500-HANDLE-INVALID.
           DISPLAY 'INVALID RECORD: ' IN-CUSTOMER-ID
                   ' ERROR: ' WS-ERROR-CODE
           ADD 1 TO WS-RECORDS-ERROR
           CALL 'ERRLOG' USING IN-CUSTOMER-ID
                               WS-ERROR-CODE.
      *
       8000-WRITE-SUMMARY.
           DISPLAY 'PROCESSING SUMMARY'
           DISPLAY '  RECORDS READ:    ' WS-RECORDS-READ
           DISPLAY '  RECORDS WRITTEN: ' WS-RECORDS-WRITTEN
           DISPLAY '  RECORDS ERROR:   ' WS-RECORDS-ERROR.
      *
       9000-TERMINATE.
           CLOSE INPUT-FILE
           IF WS-INPUT-STATUS NOT = '00'
               DISPLAY 'ERROR CLOSING INPUT FILE: ' WS-INPUT-STATUS
           END-IF
           CLOSE OUTPUT-FILE
           IF WS-OUTPUT-STATUS NOT = '00'
               DISPLAY 'ERROR CLOSING OUTPUT FILE: ' WS-OUTPUT-STATUS
           END-IF.
