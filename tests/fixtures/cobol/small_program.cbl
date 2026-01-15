       IDENTIFICATION DIVISION.
       PROGRAM-ID. SMALLPGM.
       AUTHOR. TEST.
      *
      * A small test COBOL program for unit testing.
      * Demonstrates basic structure with all four divisions.
      *
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-PC.
       OBJECT-COMPUTER. IBM-PC.
      *
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-COUNTER        PIC 9(4) VALUE ZEROS.
       01  WS-MESSAGE        PIC X(50) VALUE SPACES.
       01  WS-STATUS         PIC X(2) VALUE '00'.
      *
       PROCEDURE DIVISION.
       0000-MAIN-PROCEDURE.
           PERFORM 1000-INITIALIZE
           PERFORM 2000-PROCESS
           PERFORM 9000-TERMINATE
           STOP RUN.
      *
       1000-INITIALIZE.
           MOVE 'PROGRAM STARTED' TO WS-MESSAGE
           DISPLAY WS-MESSAGE.
      *
       2000-PROCESS.
           ADD 1 TO WS-COUNTER
           IF WS-COUNTER > 100
               MOVE 'COUNTER OVERFLOW' TO WS-MESSAGE
               MOVE '99' TO WS-STATUS
           END-IF.
      *
       9000-TERMINATE.
           MOVE 'PROGRAM ENDED' TO WS-MESSAGE
           DISPLAY WS-MESSAGE.
