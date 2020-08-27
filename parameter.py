''' Output Parameter Description

#   //number   //  Parameter    //  Variable ID //  Unit&memo
0    //0   // time //          // sec
1    //401 // Reactor Power //  RP  // %
2    //404 //  SG 1&2 level Difference   //  SGLD   //  %
3    //405 //  RCP1 on/ff   //  RCP1   //  1:on, 0: off
4    //406 //  RCP2 on/ff   //  RCP1   //  1:on, 0: off
5    //407 //  RCP3 on/ff   //  RCP1   //  1:on, 0: off
6    //408 //  RCP4 on/ff   //  RCP1   //  1:on, 0: off
7    //409 //  SG1 Level (wide)   //  SGLV1   //  %
8    //410 //  SG2 Level (wide)   //  SGLV2   //  %
9    //411 //  SG1 Level (narrow)   //     //  %
10    //412 //  SG2 Level (narrow)   //     //  %
11    //413 // FW FLow (to SG1) //  FWF1    // L/sec
12    //414 // FW FLow (to SG2) //  FWF2    // L/sec
13    //415 // AFW FLow (to SG1) //      // L/sec
14    //416 // AFW FLow (to SG2) //      // L/sec
15    //417 //  PZR Pressure    //  PP  //  kg/cm2A
16    //418 //  PZR Level   //  PL  //  %
17    //419 //  RCS subcooling margin (RCS) //  RCM //  C
18    //420 //  SG1 pressure    //  SG1 // kg/cm2A
19    //421 //  SG2 pressure    //  SG2 //  kg/cm2A
'''


''' Input Description

# // variable ID    // Parameter // condition; component operation
0   //  IND   //  INDEX (Scenario number; SGTR: ~100,000, MSLB:100,000~)  // -   ***(meaningless data for training)
1   //  BRK   //Break size (1:A, 2:2A, 4:4A)  //  -
2   //  T1   //Action 1 time   //  PZR level < 28; Charging Valve open 
3   //  A1    //  Action 1 size   //  PZR level < 28; Charging Valve open (0, 50, 100)
4   //  T2   //  Action 2 time   //  PZR pressure < 156.4; PZR heater on 
5   //  A2    //  Action 2 size   //  PZR pressure < 156.4; PZR heater on (0, 50, 100)
6   //  T3   //  Action 3 time   //  PZR pressure < 125.1; SIAS actuate 
7   //  A3    //  Action 3 size   //  PZR pressure < 125.1; SIAS actuate (1:Actuate, 0:Noactuate)
8   //  T4   //  Action 4 time   //  PZR pressure < 121; RCP 1A&2A trip 
9   //  A4    //  Action 4 size   //  PZR pressure < 121; RCP 1A&2A trip (1:Trip, 0:Notrip)
10   //  T5  //  Action 5 time   //  PCS margin < 15; RCP 1A&1B&2A&2B trip
11   //  A5   //  Action 5 size   //  PCS margin < 15; RCP 1A&1B&2A&2B trip (1:Trip, 0:Notrip)
12   //  T6  //  Action 6 time (SGTR)  //  RCS temp > 297; TBV open 
13   //  T6   //  Action 6 size (SGTR)  //  RCS temp > 297; TBV open (0, 25, 50, 75, 100)
14   //  T7  //  Action 7 time (MSLB)  //  RCS temp < 288; ADV open 
15   //  A7   //Action 7 size (MSLB)  //  RCS temp < 288; ADV open  (0, 25, 50, 75, 100)
16   //  CLAS  //  Class (0: SGTR, 1: ESDE)
'''