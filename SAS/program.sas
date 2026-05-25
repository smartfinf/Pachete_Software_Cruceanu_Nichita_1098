OPTIONS VALIDVARNAME=V7; 

FILENAME REF80 '/home/u64500499/proiect/churn-bigml-80.csv'; 
FILENAME REF20 '/home/u64500499/proiect/churn-bigml-20.csv'; 

PROC IMPORT DATAFILE=REF80 DBMS=CSV OUT=WORK.churn_80_df REPLACE; GETNAMES=YES; RUN;
PROC IMPORT DATAFILE=REF20 DBMS=CSV OUT=WORK.churn_20_df REPLACE; GETNAMES=YES; RUN;

DATA WORK.orange_raw_df;
    SET WORK.churn_80_df WORK.churn_20_df;
RUN;

PROC FORMAT;
    VALUE $churn_fmt
        'False' = 'Client Retinut'
        'True'  = 'Client Pierdut';
    VALUE plan_fmt
        0 = 'Fara Plan'
        1 = 'Cu Plan';
RUN;

DATA WORK.orange_clean_df;
    SET WORK.orange_raw_df;

    IF International_plan = 'Yes' THEN Intl_Plan_Num = 1;
    ELSE Intl_Plan_Num = 0;

    IF Voice_mail_plan = 'Yes' THEN VM_Plan_Num = 1;
    ELSE VM_Plan_Num = 0;

    Total_Mins = SUM(Total_day_minutes, Total_eve_minutes, Total_night_minutes);
    Total_Charge = SUM(Total_day_charge, Total_eve_charge, Total_night_charge, Total_intl_charge);

    ARRAY charges_array(4) Total_day_charge Total_eve_charge Total_night_charge Total_intl_charge;
    DO i = 1 TO 4;
        charges_array(i) = ROUND(charges_array(i), 0.01); 
    END;
    DROP i; 

    FORMAT Intl_Plan_Num plan_fmt.;
RUN;

DATA WORK.orange_churners_df;
    SET WORK.orange_clean_df;
    WHERE Churn = 'True'; 
RUN;

PROC SQL;
    CREATE TABLE WORK.state_summary_df AS
    SELECT State,
           COUNT(*) AS Total_Customers,
           MEAN(Total_Charge) AS Avg_State_Charge
    FROM WORK.orange_clean_df
    GROUP BY State;

    CREATE TABLE WORK.orange_enriched_df AS
    SELECT a.*, b.Avg_State_Charge
    FROM WORK.orange_clean_df AS a
    LEFT JOIN WORK.state_summary_df AS b
    ON a.State = b.State;
QUIT;

PROC REPORT DATA=WORK.state_summary_df(OBS=10) HEADLINE HEADSKIP;
    TITLE "Top 10 State: Numar Clienti si Cost Mediu Total";
    COLUMN State Total_Customers Avg_State_Charge;
    DEFINE State / DISPLAY 'Statul';
    DEFINE Total_Customers / DISPLAY 'Total Clienti';
    DEFINE Avg_State_Charge / DISPLAY 'Cost Mediu ($)' FORMAT=8.2;
RUN;

PROC MEANS DATA=WORK.orange_clean_df N MEAN MIN MAX MAXDEC=2;
    TITLE "Statistici Descriptive: Costuri, Minute si Apeluri la Suport";
    VAR Total_Mins Total_Charge Customer_service_calls;
    CLASS Churn;
RUN;

PROC LOGISTIC DATA=WORK.orange_clean_df DESCENDING;
    TITLE "Model de Regresie Logistica: Probabilitatea de Parasire a Retelei";
    CLASS Intl_Plan_Num (PARAM=REF REF='Fara Plan');
    MODEL Churn = Total_Charge Customer_service_calls Intl_Plan_Num;
RUN;

PROC SGPLOT DATA=WORK.orange_clean_df;
    TITLE "Distributia Apelurilor la Customer Service in functie de Statusul Clientului";
    VBAR Customer_service_calls / GROUP=Churn GROUPDISPLAY=CLUSTER;
    XAXIS LABEL="Numar Apeluri Serviciul Clienti";
    YAXIS LABEL="Frecventa";
RUN;

PROC SGPLOT DATA=WORK.orange_clean_df;
    TITLE "Cost Total vs. Minute Totale Consumate";
    SCATTER X=Total_Mins Y=Total_Charge / GROUP=Churn;
RUN;