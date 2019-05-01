#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
import os


# In[ ]:


def csv2txt(csv):
    file = []
    with open(csv, 'r') as f: 
        for line in f.readlines():
            file.append(line)
    mx = 0
    key = 0
    for i in file:
        if ('Key_signature' in i):
            key = int(i.split(',')[3])
        if ('End_track' in i):
            length = int(i.split(',')[1])
            break
    everything = np.zeros((16, int(length/256) + 1))
    active = np.zeros(16)
    previous = (0,0,0,0,0,0)
    for i in file[8: -2]:
        if ('Key_signature' in i):
            key = int(i.split(',')[3].strip())
        if (len(i.split(',')) == 6):
            track, time, on, channel, note, velocity = i.strip().split(',')
            time = int(int(time)/256)
            track = int(track)
            channel = int(channel)
            note = int(note) 
            note -= int(key)
            mx = max(mx, note)
            velocity = int(velocity)
            vec = (track, time, on, channel, note, velocity)
            if (on == ' Note_off_c'):
                for i in range(np.int(active[channel]), time + 1):
                    everything[channel, i] = note
            elif (on == ' Note_on_c'):
                active[channel] = time
            previous = vec
    everything = np.delete(everything, np.argwhere(everything.sum(axis = 1) == 0), axis = 0)
    st = ''
    for i in everything.T:
        for j in i:
            if (j != 0):
                st += chr(int(j) + 34)
            else:
                st += chr(34)
        st += ' '
    return st


# In[3]:


def str2csv(st, f_name):
    if(st[-1] != ' '):
        st += ' '
    noc = 0
    for inst in st.split(' '):
        noc = max(len(inst), noc)
    noc = (min(noc, 15))
    song = np.ones((noc, st.count(' ')))*34
    count = 0
    for inst in st.split(' '):
        count2 = 0
        for note in inst:
            song[count2, count] = ord(note) 
            count2 +=1
        count += 1
    file = ['0, 0, Header, 1, %d, 1024\n' % (song.shape[0] + 1), 
        '1, 0, Start_track\n',
        '1, 0, Time_signature, 4, 2, 24, 8\n',
        '1, 0, Key_signature, 0, "major"\n',
        '1, 0, Tempo, 750000\n',
        '1, ' + str(count * 256)  + ', End_track\n']
    for i in range(song.shape[0]):
        curr = 34
        file.append("%d, %d, %s\n" % (i+2, 0, "Start_track"))
        file.append("%d, %d, %s, %s %d\n" % (i+2, 0, "Title_t", "Instrument", i + 1))
        file.append("%d, 0, Control_c, 0, 7, 101\n" % (i+2))
        file.append("%d, 0, Control_c, 0, 10, 64\n" % (i+2))
        for j in range(song.shape[1]):

            if (song[i][j] != curr):
                if (curr != 34):   
                    file.append("%d, %d, %s, %d, %d, %d\n" % (i + 2, j * 256, 'Note_off_c', i, curr - 34, 0))
                file.append("%d, %d, %s, %d, %d, %d\n" % (i + 2, j * 256, 'Note_on_c', i, song[i][j] - 34, 96))
                curr = song[i][j]

        file.append("%d, %d, %s\n" % (i+2, (song.shape[1] + 1) * 256, "End_track"))
    file.append("0, 0, End_of_file")
    f = open(f_name, "w")
    for i in file:
        f.write(i)    
    f.close()


# In[191]:


f = open("complete_bach.txt", "w")   
for filename in os.listdir("datac"):
    if (filename.endswith(".csv")):
        f.write(csv2txt("datac/" + filename))
        f.write('\n')
f.close()


# In[7]:


str2csv("Whka hc_\\O da]\\ da]Z da\\X da\\X da\\Z da\\N kc\\P kc\P md\\P kd\\P kd\\P kd\\P kd\\P kd\\P fd]P dd\\Q dd]Q d]ZN d ", "create.csv")


# In[8]:


str2csv("Wh` h`WU hcYS hcZS hcZS haZS haXQ haXQ haXQ h_XP h_ZW h_XX h_XX h_XX h_XX h_XX h_XX h_XX h_XX j_XX jaX ", "create2.csv")


# In[9]:


str2csv("Wh hbXUH haYUu haYU haYU haYU haYU haYU hdaU hd_U hdaU hdaU hdaU faWS faWK faXU faXS faXQ faXQ faXQ fa ", "create3.csv")


# In[13]:


str2csv('+YWW _\WW c_\W c`XW c`WW `cYY c`YY b`YY b`YY eb^Y eb^Y gc`[ gc`[ gc`[ gb`[ o``[ oc`[ qc`Y me`Y qe`Y qe`Y qe`Y qe`Y le`Y e`]Y ^^YR g^WT ^[W[ ^[W[ `^[W `[[X `][X `^[X `]ZY c`[X c`[Y eb^T eb^V eb^V eb^V eb^V eb^V eb^V eb^V eb^V gc^V gb^ gl^ nb^ lg^T le[T le[T le]T le]T le]T qe]T ee]T gc]T gc^T eb]T gb[R b[[R b[[" b[[" b["[ b[[" b]]" kb]" bg[" kb[" kbY" i`W" lgW" lcT" lcT" lc"Q le`P leb^ leb^ leb^ lee', "create4.csv")
        


# In[6]:


str2csv('Thd_SG fc_SGfc fc]SffS dc]QdcZ dd\PPfc d__PPd_ d__Pd_P d_aQ_d_ d__P\dd f__Sfd_ f__S_f_ f__SSfi_ if_XSif_ kf_SSkfS kf_SSkf_ kf_SXkdS kf_WWffS kf_WWkS d_\XXkd _\XXk\Z i`ZWWii i_WWWic i_WWWic __WW__ _ZWWS _ZWSW _ZWSS _ZWSG _ZWSSG _ZWSS_Z h_XXL_h h_X\Xhh h_WWWh_ h_WWWK h_WWWP hc_WUI "daUUI "mdUUI "mdUUI kdaUR jdaUd """XR" """ZXQ """ZQ" ""ZX"" ""X"" ""X"" "X""" ""X"" "X""" ""_"" "f""S" "fa"S" "da""U hda"', "create5.csv")


# In[15]:


str2csv('Th^[T ha^YF iaYN iaZN icZS hcZS hcZW hdaX hdaU hdaU hdaU fc_S fc_S if_X fg_X hf_X h__X h`_X h`aW ha^X ja^\\ "^"^ "a^" a^^= ^[K" ]"K "_R "YS "YY "cW "^W "_W "\\U "^U "cT "aU c]U" cWT" ]ZW" _ZW" "\\X" "Z"" a]"" a_"" da"Q dd"P fd"Q ffUI ddUI fkSG kfNB kkhS kdX" kdX" mdX" jdX" mbX" kdX" kdX" khX" khL" jkZ" iiZ" kiZ_ ii`Z ihbZ iffZ if`Z if`Z ifbZ kb"X k`]W ka]U ea]S ec]S kc]S kc\\S kd\\X kd\\X kd\\X kg]X jf]V lg]W lg[W lg[W lf]V mf]V lf]V kc]V kc]W kfbI jgbN jgjN ne ncN neYN neYF neYF neYnF neYFl neYF teYY le[Y leYY neYY ngYY lg[S gg[T gg]T gg]T gg`T ge`T ie`T i`iY i`Y i`V i`V jgV jdV jgc vjc oj` oi` li`" l', "create6.csv")


# In[16]:


str2csv('9; ph_XL@ ph_XL@ ph_XL@ ph_XL@ "h_XL@ "_hXL@ "hdXL@ "hdXL@ "hdXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ "daXL@ hd_XL@ h_X_L@ h__XL@ h__XL@ "__XL@ "__XL@ "__XL@ "_X_L@ "_\XL@ "_\XL@ h_XXL@ "_XL@ "_XL@ "_XL@ "_XL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ "XXL@ ]XXL', "create7.csv")


# In[5]:


str2csv('\nh_W hd_X hd_X hd_X hd_X id]Z id]Z idaZ idaZ idaZ idaZ idaZ jdaZ jebZ jdaZ jdaZ jdaZ jdaZ jdaZ icZZ hcZS hcZS haZU ha\\U ha]Y ha\\Y ia\\Y ia]Z ia]Z ia]Z ia]Z iaa] iaa] iaaZ icaZ jbZZ jjZN jcZN jcZN jcZN jaXK iaWK iaXN haXN h_XP h_XP h_XL h_XL h_XL h_\\X h_\\X h_\\X h_\\X h_\\X h_\\X h_\\X h_\\X h_\\X h_\\X h_ZX h_XX h_XX h_XX h_XX hc_X hc_X hc_X \nd_\\X d_\\X d_\\X d_\\W d_\\U d_\\U d_\\U d_\\U d_\\U d_\\U d_\\U d_\\U d\\UU', "LSTMMUSIC2.csv")
        


# In[ ]:




