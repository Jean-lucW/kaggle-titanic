# Findings text for tickets_v1.ipynb, read by build_tickets_v1.py. Numbers are from the run on complete_data.csv.

FINDINGS = {
"summary": """## Findings

- **Ticket type mostly restates class and port.** PC tickets are all 1st class and 78% Cherbourg. A/5, A/4, SOTON/O.Q., STON/O 2. and C are all 3rd class and almost all Southampton. SC/PARIS is 2nd class Cherbourg. But 73% of passengers hold a number-only ticket, so type alone guesses class right for 65% of passengers, against 54% with no ticket information.
- **The ticket number says far more than the type.** For 6-digit number-only tickets the first digit is the class: every 1xxxxx is 1st class, every 2xxxxx 2nd, every 3xxxxx 3rd. Across all tickets, the number band guesses class right for 96% of passengers and the window for 98%.
- **Windows also pick out port.** All 90 passengers on 26xx tickets are 3rd class from Cherbourg. 330xxx, 367xxx, 368xxx and 382xxx are entirely Queenstown. Within PC, the Cherbourg share rises from 38% in PC 174xx to 96% in PC 177xx. Port is guessed right for 88% of passengers from the window, 75% from the type.
- **Fare follows the window.** The fare per person band (fifths) is guessed right for 74% of passengers from the window, 38% from the type, 21% with no ticket information.
- **Deck a little, sex not at all.** Some 1st class windows sit on one deck (19xxx: 94% deck C; PC 176xx: 88% deck B), lifting deck guesses from 32% to 48%. The ticket tells nothing about sex.
- **Survival beyond class and sex is weak.** The window guesses survival right for 68% of train passengers against 62% with no information, almost all of it through class. A handful of windows sit 20 points or more from what their class and sex predict, but each holds 8 to 18 train passengers.""",

"how": """Class, port and fare are where the ticket carries real information, and the number does most of the work. Cabin recorded and survival improve from type to band (81% to 90%, 64% to 69%) because both follow class; the window adds nothing further. Sex scores below the no-information guess at every level: the ticket carries nothing about it, and leaving out the passenger's own value costs a little in small groups.""",

"type": """Every prefixed type except C.A., W./C. and Other prefixes sits in a single class, and all but PC and SC/PARIS are almost entirely Southampton. Cabins are recorded mainly on PC tickets (73%) and on number-only tickets (22%, mostly 1st class). The survival differences between types are mostly class: PC (1st class) survived at 65%, A/5, A/4 and W./C. (3rd class) at 0 to 10%.""",

"fare": """Fare per person separates class cleanly within a type: 1st class above 20 GBP, 2nd class 10 to 15, 3rd class under 10. PC tickets cost more than number-only 1st class tickets (median 35.64 against 27.95 per person). Among 3rd class types, C.A. was cheapest (median 6.32) and SOTON/O.Q. 7.15, against 7.75 for number-only 3rd class. Fare adds little that class does not already say.""",

"scatter": """The number-only tickets fall into distinct blocks. 4-digit tickets are almost all 3rd class, and the 2,000 to 3,000 block is Cherbourg. 5-digit tickets from 11,000 to 20,000 are mostly 1st class, 28,000 to 30,000 2nd class. 6-digit tickets split exactly by first digit into 1st, 2nd and 3rd class, with the 3rd class block around 330,000 to 384,000 holding most Queenstown passengers. The 7-digit block at 3.1M is 3rd class Southampton, matching the STON/O 2. and SOTON/O.Q. tickets in the same range.""",

"band": """The band alone is enough for class on most tickets: 1xxxxx, 2xxxxx and 3xxxxx are each 100% one class, 1xxxx is 92% 1st class and 2xxxx 84% 2nd class. 2xxx is 93% 3rd class and 93% Cherbourg. The mixed bands are 3xxxx (56% 1st, 16% 2nd, 28% 3rd) and the small 3- and 5-digit bands. Port needs the finer window: 3xxxxx is 72% Southampton and 25% Queenstown overall.""",

"win": """Inside each band, windows sort passengers further:

- **Port.** 26xx is 100% Cherbourg. 330xxx, 367xxx, 368xxx and 382xxx are 100% Queenstown, 370xxx 81% and 371xxx 71%. 345xxx, 347xxx, 349xxx and 350xxx, the largest 3rd class windows, are Southampton.
- **Deck.** Among 1st class windows, 19xxx is 94% deck C, 16xxx 83% deck E, 112xxx 71% deck B. 113xxx, the largest 1st class window (66 passengers), spreads over A to E.
- **Fare.** 239xxx is 9 2nd class passengers with a fare of 0; all 8 in train died, all male. 382xxx has the lowest median fare per person, 4.85.
- **Mixed windows.** 35xxx and 36xxx are the only windows mixing 1st and 3rd class, and they also mix all three ports.""",

"pre": """The number resolves the class split inside C.A. and W./C.: C.A. 21xx, 23xx, 26xx and 37xxx are all 3rd class, C.A. 29xxx to 34xxx all 2nd class; W./C. 66xx is 3rd class, W./C. 14xxx 2nd. Within PC, the Cherbourg share rises with the number (38%, 77%, 86%, 96%), and PC 176xx is 88% deck B. SOTON/O.Q. 310xxxx and STON/O 2. 310xxxx share a number range and a class, yet none of the 8 SOTON/O.Q. passengers in train survived against 44% of the 18 STON/O 2. passengers.""",

"surv": """Most windows sit within 15 points of the rate their class and sex predict, so the ticket adds little to survival once class and sex are known. The largest gaps are above expected for 367xxx (67% against 30%, 9 passengers), 230xxx (88% against 54%, 8), PC 174xx (90% against 61%, 10) and STON/O 2. 310xxxx (44% against 24%, 18), and below for 41xx (22% against 46%, 9, 8 of them female). These windows are mostly a few families on consecutive tickets, so the gap is likely the family sharing one fate rather than the ticket range. The most solid gaps come from the large windows: 349xxx (53 in train) survived at 6% against 19% expected, 26xx (57) at 40% against 28%.""",
}
