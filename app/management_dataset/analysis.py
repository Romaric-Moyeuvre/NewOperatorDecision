## IMPORT ##########

####################

def get_stats(matrix) : 
    number_of_measurements = len(matrix[0])

    # P S Smin Smed Smax T Tmin Tmed Tmax 

    if number_of_measurements == 1 :

        P = matrix[0][0]
        S = matrix[1][0]
        Smin = matrix[2][0]
        Smed = matrix[3][0]
        Smax = matrix[4][0]
        T = matrix[5][0]
        Tmin = matrix[6][0]
        Tmed = matrix[7][0]
        Tmax = matrix[8][0]

        abs_S_Smin = abs(S-Smin)
        rel_S_Smin_semi_width = abs(S-Smin)/abs(Smed-Smin) if abs(Smed-Smin) != 0 else abs_S_Smin
        rel_S_Smin_full_width = abs(S-Smin)/abs(Smax-Smin) if abs(Smax-Smin) != 0 else abs_S_Smin
        count_anomalies_S = 1 if S < Smin or S > Smax else 0
        ratio_anomalies_S = 1 if S < Smin or S > Smax else 0
        max_variation_S = 0

        abs_T_Tmin = abs(T-Tmin)
        rel_T_Tmin_semi_width = abs(T-Tmin)/abs(Tmed-Tmin) if abs(Tmed-Tmin) != 0 else abs_T_Tmin
        rel_T_Tmin_full_width = abs(T-Tmin)/abs(Tmax-Tmin) if abs(Tmax-Tmin) != 0 else abs_T_Tmin
        count_anomalies_T = 1 if (T < Tmin or T > Tmax) else 0
        ratio_anomalies_T = 1 if (T < Tmin or T > Tmax) else 0
        max_variation_T = 0

        mean_correlation = 0
        nb_measurements = number_of_measurements

    else : 

        abs_S_Smin = 0
        rel_S_Smin_semi_width = 0
        rel_S_Smin_full_width = 0
        count_anomalies_S = 0

        abs_T_Tmin = 0
        rel_T_Tmin_semi_width = 0
        rel_T_Tmin_full_width = 0
        count_anomalies_T = 0

        nb_measurements = number_of_measurements

        max_variation_S = 0
        max_variation_T = 0
        mean_correlation = 0

        for k in range(number_of_measurements) :

            P = matrix[0][k]
            S = matrix[1][k]
            Smin = matrix[2][k]
            Smed = matrix[3][k]
            Smax = matrix[4][k]
            T = matrix[5][k]
            Tmin = matrix[6][k]
            Tmed = matrix[7][k]
            Tmax = matrix[8][k]

            abs_S_Smin = max(abs(S-Smin), abs_S_Smin)
            rel_S_Smin_semi_width = max(abs(S-Smin)/abs(Smed-Smin), rel_S_Smin_semi_width) if abs(Smed-Smin) != 0 else rel_S_Smin_semi_width
            rel_S_Smin_full_width = max(abs(S-Smin)/abs(Smax-Smin), rel_S_Smin_full_width) if abs(Smax-Smin) != 0 else rel_S_Smin_full_width
            count_anomalies_S += 1 if S < Smin or S > Smax else 0

            abs_T_Tmin = max(abs(T-Tmin), abs_T_Tmin)
            rel_T_Tmin_semi_width = max(abs(T-Tmin)/abs(Tmed-Tmin), rel_T_Tmin_semi_width) if abs(Tmed-Tmin) != 0 else rel_T_Tmin_semi_width
            rel_T_Tmin_full_width = max(abs(T-Tmin)/abs(Tmax-Tmin), rel_T_Tmin_full_width) if abs(Tmax-Tmin) != 0 else rel_T_Tmin_full_width
            count_anomalies_T += 1 if T < Tmin or T > Tmax else 0

        for k in range(number_of_measurements-1) :
            P_k = matrix[0][k]
            S_k = matrix[1][k]
            T_k = matrix[5][k]
            P_k2 = matrix[0][k+1]
            S_k2 = matrix[1][k+1]
            T_k2 = matrix[5][k+1]

            max_variation_S = max((abs(S_k-S_k2)*abs(P_k+P_k2))/(abs(S_k+S_k2)*abs(P_k-P_k2)), max_variation_S) if (abs(S_k+S_k2)*abs(P_k-P_k2)) else max_variation_S
            max_variation_T = max((abs(T_k-T_k2)*abs(P_k+P_k2))/(abs(T_k+T_k2)*abs(P_k-P_k2)), max_variation_T) if (abs(T_k+T_k2)*abs(P_k-P_k2)) else max_variation_T
            mean_correlation += abs(S_k*T_k2-S_k2*T_k)/number_of_measurements

        ratio_anomalies_S = count_anomalies_S / number_of_measurements
        ratio_anomalies_T = count_anomalies_T / number_of_measurements

    S_tab = [abs_S_Smin,rel_S_Smin_semi_width,rel_S_Smin_full_width,count_anomalies_S,ratio_anomalies_S,max_variation_S]
    T_tab = [abs_T_Tmin,rel_T_Tmin_semi_width,rel_T_Tmin_full_width,count_anomalies_T,ratio_anomalies_T,max_variation_T]
    B_tab = [mean_correlation,nb_measurements]

    return S_tab+T_tab+B_tab