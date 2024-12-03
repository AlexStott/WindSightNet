

import emd
import pywt


import os
import math
from math import floor
import time
import numpy as np

import pandas as pd  
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import pickle
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.units as munits
import datetime
import matplotlib.pyplot as plt

from scipy.io import savemat

from matplotlib.dates import date2num
import re

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from obspy import read_inventory, UTCDateTime
import matplotlib.dates as mdates

import sys
from pathlib import Path

import seaborn as sns

import plotly.express as px

from scipy.stats import pearsonr
global configfile


print("libraries imported")

#%% Mars time converter
#configfile = './landerconfig.xml'
path2marsconverter = '/Users/a.stott/marsconverter'  #os.environ['MARSCONVERTER']

# CONFIG FILE FOR MER A - SPIRIT ROVER
# configfile = path2marsconverter+'/landerconfigSpirit.xml'

# CONFIG FILE FOR INSIGHT LANDER
configfile = path2marsconverter+'/landerconfig.xml'

class MarsConverter:
    """
    Class which contains all the function to convert UTC time to Martian Time.
    All the calculation are based of the Mars 24 algorithm itself based on
    Alison, McEwen, Planetary ans Space Science 48 (2000) 215-235
    https://www.giss.nasa.gov/tools/mars24/help/algorithm.html
    """

    # JULIAN UNIX EPOCH (01/01/1970-00:00 UTC)
    JULIAN_UNIX_EPOCH = 2440587.5

    # MILLISECONDS IN A DAY
    MILLISECONDS_IN_A_DAY = 86400000.0

    # SECOND IN A DAY
    SECOND_IN_A_DAY = 86400.0

    # SHIFT IN DAY PER DEGREE
    SHIFT_IN_DAY_PER_DEGREE = 1. / 360.0

    # Julian day of reference (january 2010, 6th 00:00 UTC) At that time,
    # the martian meridian is also at midnight.
    CONSISTENT_JULIAN_DAY = 2451545.0

    # Delta between IAT and UTC
    TAI_UTC = 0.0003725

    # Time division
    TIME_COEFFICIENT = 60.0

    # Millisecond multiplier
    MMULTIPLIER = 1000.0

    # Allison's coefficient
    K = 0.0009626

    # Allison's normalisation factor to make sure that we get positive values
    # for date after 1873
    KNORM = 44796.0

    # Ratio Betwwen Martian SOL and terrestrial day
    SOL_RATIO = 1.0274912517

    LONGITUDE = None

    # Sol-001 and Sol-002 start times to compute one Martian day in seconds.
    # Cannot use landing time because Sol-000 lasted shorter.
    SOL01_START_TIME = UTCDateTime("2018-11-27T05:50:25.580014Z")
    SOL02_START_TIME = UTCDateTime("2018-11-28T06:30:00.823990Z")

    SECONDS_PER_MARS_DAY = SOL02_START_TIME - SOL01_START_TIME - 0.000005

    def __init__(self, landerconfigfile=None):

        global configfile
        from lxml import etree

        if landerconfigfile is not None:
            configfile = configfile
        else:
            configfile = configfile

        tree = etree.parse(configfile)
        root = tree.getroot()
        for el in root:
            if el.tag == 'landingdate':
                LANDING_DATE_STR = el.text
            if el.tag == 'longitude':
                LANDING_LONGITUDE = float(el.text)
            if el.tag == 'solorigin':
                SOL_ORIGIN_STR = el.text
            if el.tag == 'latitude':
                LANDING_LATITUDE = float(el.text)

        utc_origin_sol_date = UTCDateTime(SOL_ORIGIN_STR)
        utc_landing_date = UTCDateTime(LANDING_DATE_STR)

        self.__landingdate = utc_landing_date
        self.__longitude = LANDING_LONGITUDE
        self.__origindate = utc_origin_sol_date
        self.LONGITUDE = float(self.__longitude)
        self.LATITUDE = LANDING_LATITUDE

    def get_landing_date(self):
        """
        Returns the landing date of the lander in UTCDateTime format
        """
        return self.__landingdate

    def get_longitude(self):
        """
        Returns the lander longitude
        """
        return self.__longitude

    def get_origindate(self):
        return self.__origindate

    def j2000_epoch(self):
        """
        Returns the j2000 epoch as a float
        """
        return self.CONSISTENT_JULIAN_DAY

    def mills(self):
        """
        Returns the current time in milliseconds since Jan 1 1970
        """
        return time.time()*self.MMULTIPLIER

    def julian(self, date=None):
        """
        Returns the julian day number given milliseconds since Jan 1 1970
        """
        if date is None:
            dateUTC = UTCDateTime.now()
        else:
            dateUTC = UTCDateTime(date)
            millis = dateUTC.timestamp * 1000.0

        return self.JULIAN_UNIX_EPOCH + (millis/self.MILLISECONDS_IN_A_DAY)

    def utc_to_tt_offset(self, jday=None):
        """
            Returns the offset in seconds from a julian date in
            Terrestrial Time (TT) to a Julian day in Coordinated
            Universal Time (UTC)
        """
        return self.utc_to_tt_offset_math(jday)

    def utc_to_tt_offset_math(self, jday=None):
        """
        Returns the offset in seconds from a julian date in
        Terrestrial Time (TT) to a Julian day in Coordinated
        Universal Time (UTC)
        """
        if jday is None:
            jday_np = self.julian()
        else:
            jday_np = jday

        jday_min = 2441317.5
        jday_vals = [-2441317.5, 0, 182, 366, 731, 1096, 1461, 1827,
                     2192, 2557, 2922, 3469, 3834, 4199, 4930, 5844,
                     6575, 6940, 7487, 7852, 8217, 8766, 9313, 9862,
                     12419, 13515, 14792, 15887, 16437]

        offset_min = 32.184
        offset_vals = [-32.184, 10.0, 11.0, 12.0, 13.0,
                       14.0, 15.0, 16.0, 17.0, 18.0,
                       19.0, 20.0, 21.0, 22.0, 23.0,
                       24.0, 25.0, 26.0, 27.0, 28.0,
                       29.0, 30.0, 31.0, 32.0, 33.0,
                       34.0, 35.0, 36.0, 37.0]

        if jday_np <= jday_min+jday_vals[0]:
            return offset_min+offset_vals[0]
        elif jday_np >= jday_min+jday_vals[-1]:
            return offset_min+offset_vals[-1]
        else:
            for i in range(0, len(offset_vals)):
                if (jday_min+jday_vals[i] <= jday_np) and \
                        (jday_min+jday_vals[i+1] > jday_np):
                    break
            return offset_min+offset_vals[i]

    def julian_tt(self, jday_utc=None):
        """
        Returns the TT Julian Day given a UTC Julian day
        """
        if jday_utc is None:
            jday_utc = self.julian()
        jdtt = jday_utc + self.utc_to_tt_offset(jday_utc)/86400.
        return jdtt

    def j2000_offset_tt(self, jday_tt=None):
        """
        Returns the julian day offset since the J2000 epoch
        (AM2000, eq. 15)
        """
        if jday_tt is None:
            jday_tt = self.julian_tt()
        return (jday_tt - self.j2000_epoch())

    def Mars_Mean_Anomaly(self, j2000_ott=None):
        """
        Calculates the Mars Mean Anomaly for a givent J2000 julien 
        day offset (AM2000, eq. 16)
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()
        M = 19.3871 + 0.52402073 * j2000_ott
        return M % 360.

    def Alpha_FMS(self, j2000_ott=None):
        """
        Returns the Fictional Mean Sun angle
        (AM2000, eq. 17)
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()

        alpha_fms = 270.3871 + 0.524038496 * j2000_ott

        return alpha_fms % 360.

    def alpha_perturbs(self, j2000_ott=None):
        """
        Returns the perturbations to apply to the FMS Angle from orbital
        perturbations. (AM2000, eq. 18)
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()

        array_A = [0.0071, 0.0057, 0.0039, 0.0037, 0.0021, 0.0020, 0.0018]
        array_tau = [2.2353, 2.7543, 1.1177, 15.7866, 2.1354, 2.4694, 32.8493]
        array_phi = [49.409, 168.173, 191.837, 21.736, 15.704, 95.528, 49.095]

        pbs = 0
        for (A, tau, phi) in zip(array_A, array_tau, array_phi):
            pbs += A*np.cos(((0.985626 * j2000_ott/tau) + phi)*np.pi/180.)
        return pbs

    def equation_of_center(self, j2000_ott=None):
        """
        The true anomaly (v) - the Mean anomaly (M)
        (Bracketed term in AM2000, eqs. 19 and 20)
        Section B-4. on the website

        ----
        INPUT
            @j2000_ott: float - offseted terrestrial time relative to j2000
        ----
        OUTPUT
            @return: EOC
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()

        M = self.Mars_Mean_Anomaly(j2000_ott)*np.pi/180.
        pbs = self.alpha_perturbs(j2000_ott)

        EOC = (10.691 + 3.0e-7 * j2000_ott)*np.sin(M)\
            + 0.6230 * np.sin(2*M)\
            + 0.0500 * np.sin(3*M)\
            + 0.0050 * np.sin(4*M)\
            + 0.0005 * np.sin(5*M) \
            + pbs

        return EOC

    def L_s(self, j2000_ott=None):
        """
        Returns the Areocentric solar longitude (aka Ls)
        (AM2000, eq. 19)
        Section B-5. 
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()

        alpha = self.Alpha_FMS(j2000_ott)
        v_m = self.equation_of_center(j2000_ott)

        ls = (alpha + v_m)
        ls = ls % 360
        return ls

    def get_utc_2_ls(self, utc_date=None):
        """
        Convert UTC date to aerocentric solar longitude (Ls).
        ----
        INPUT:
            @utc_date: UTCDateTime
        ----
        OUTPUT:
            @ls : float
        """
        if not utc_date:
            utc_date = UTCDateTime().now()
        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)
        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)
        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)
        ls = self.L_s(j2000_ott=jd_ott)
        return ls

    def equation_of_time(self, j2000_ott=None):
        """
        Equation of Time, to convert between Local Mean Solar Time
        and Local True Solar Time, and make pretty analemma plots
        (AM2000, eq. 20)
        """
        if j2000_ott is None:
            j2000_ott = self.j2000_offset_tt()

        ls = self.L_s(j2000_ott)*np.pi/180.

        EOT = 2.861*np.sin(2*ls)\
            - 0.071 * np.sin(4*ls)\
            + 0.002 * np.sin(6*ls) - self.equation_of_center(j2000_ott)
        return EOT

    def get_utc_2_eot(self, utc_date=None):
        """
        Getter function to evaluation EOT for a given
        UTC time.
        ----
        INPUT:
            utc_date : UTCDateTime format date time


        """
        if utc_date is None:
            utc_date = UTCDateTime().now()

        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)
        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)

        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)
        eot = self.equation_of_time(j2000_ott=jd_ott)
        return eot

    def j2000_from_Mars_Solar_Date(self, msd=0):
        """
        Returns j2000 based on MSD
        """
        j2000_ott = ((msd + 0.00096 - 44796.0) * 1.027491252) + 4.5
        return j2000_ott

    def j2000_ott_from_Mars_Solar_Date(self, msd=0):
        """
        Returns j2000 offset based on MSD

        """
        j2000 = self.j2000_from_Mars_Solar_Date(msd)
        j2000_ott = self.julian_tt(j2000+self.j2000_epoch())
        return j2000_ott-self.j2000_epoch()

    def Mars_Solar_Date(self, j2000_ott=None):
        """
        Return the Mars Solar date
        """
        if j2000_ott is None:
            jday_tt = self.julian_tt()
            j2000_ott = self.j2000_offset_tt(jday_tt)
        const = 4.5
        MSD = (((j2000_ott - const)/self.SOL_RATIO) + self.KNORM - self.K)
        return MSD

    def Coordinated_Mars_Time(self, j2000_ott=None):
        """
        The Mean Solar Time at the Prime Meridian
        (AM2000, eq. 22, modified)
        Be aware that the correct version of MTC should be
        MTC%24 but since we need to reverse the equations to go from lmst to
        utc, we decided to apply the modulo, later.
        """
        if j2000_ott is None:
            jday_tt = self.julian_tt()
            j2000_ott = self.j2000_offset_tt(jday_tt)
        MTC = 24 * (((j2000_ott - 4.5)/self.SOL_RATIO) + self.KNORM - self.K)
        return MTC

    def j2000_tt_from_CMT(self, MTC=None):
        """
        Estimate j2000_ott from Coordinated Mars Time
        from (AM2000, eq. 22, modified)
        """
        j2000_ott = (((MTC / 24.)+self.K - self.KNORM) * self.SOL_RATIO) + 4.5
        return j2000_ott

    def _LMST(self, longitude=0, j2000_ott=None):
        """
        The Local Mean Solar Time given a planetographic longitude
        19-03-12 : modif: the modulo 24 of MTC is estimated here
        (C-3)
        """
        if j2000_ott is None:
            jday_tt = self.julian_tt()
            j2000_ott = self.j2000_offset_tt(jday_tt)
        MTC = self.Coordinated_Mars_Time(j2000_ott)
        MTCmod = MTC % 24
        # MTCmod = MTC
        LMST = (MTCmod - longitude * (24./360.)) % 24
        return LMST

    def _LTST(self, longitude=0, j2000_ott=None, mod24=True):
        """
        Local true solar time is the Mean solar time + equation of
        time perturbation from (AM2000, Eq. 23 & Eq. 24)

        """
        if j2000_ott is None:
            jday_tt = self.julian_tt()
            j2000_ott = self.j2000_offset_tt(jday_tt)

        eot = self.equation_of_time(j2000_ott)
        lmst = self._LMST(longitude, j2000_ott)
        if mod24:
            ltst = (lmst + eot*(1./15.)) % 24
        else:
            ltst = (lmst + eot*(1./15.))
        return ltst


    # ------------------------------------------------------------------------
    #     LTST : Local True Solar Time
    
    def get_utc_2_ltst(self, utc_date=None, output="date"):
        """
        Convert UTC date to LTST date.
        ----
        INPUT:
            @utc_date: UTCDateTime
            @output: str - specify the output format (decimal or date)

        ----
        OUTPUT:
            @lmst_date : str
        """
        if utc_date is None:
            utc_date = UTCDateTime().now()

        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)

        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)

        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)

        origin_in_sec = self.__origindate.timestamp
        any_date_in_sec = utc_date.timestamp

        delta_sec = any_date_in_sec - origin_in_sec
        raw_martian_sol = delta_sec / (self.SECONDS_PER_MARS_DAY)

        nb_sol = floor(math.modf(raw_martian_sol)[1])

        ltst_date = self._LTST(longitude=self.__longitude,
                               j2000_ott=jd_ott,
                               mod24=False)

        if ltst_date < 0:
            nb_sol -= 1
        elif ltst_date >= 24:
            nb_sol += 1
        ltst_date = ltst_date % 24

        ihour = int(math.modf(ltst_date)[1])
        min_dec = 60*(math.modf(ltst_date)[0])
        iminutes = floor(min_dec)
        seconds = (min_dec - iminutes)*60.
        iseconds = int(math.modf((seconds))[1])
        milliseconds = int(math.modf((seconds-iseconds))[0]*1000000)

        if output == "decimal":
            ltst_str = ltst_date
        else:
            ltst_str = "{:04}T{:02}:{:02}:{:02}.{:06}".format(nb_sol,
                                                          ihour,
                                                          iminutes,
                                                          iseconds,
                                                          milliseconds)
        return ltst_str

    def utcDateTime_to_jdutc(self, date=None):
        """
        Function to convert UTCDateTime to Julian date

        """
        if date is None:
            date = UTCDateTime().now()

        millis = date.timestamp * 1000.0
        jd_utc = self.JULIAN_UNIX_EPOCH + \
            (float(millis) / self.MILLISECONDS_IN_A_DAY)
        return jd_utc

    def jdutc_to_UTCDateTime(self, jd_utc=None):
        """
        Function to convert Julien date to UTCDateTime
        """

        millis = (jd_utc - self.JULIAN_UNIX_EPOCH) * self.MILLISECONDS_IN_A_DAY
        utc_tstamp = millis/1000.
        return UTCDateTime(utc_tstamp)

    def get_utc_2_lmst(self, utc_date=None, output="date"):
        """
        Convert UTC date to LMST date.
        Output is formated with SSSSTHH:MM:ss.mmmmmmm if output is 'date'
        Otherwise, the output is float number if output is 'decimal'
        ----
        INPUT:
            @utc_date: UTCDateTime
            @output: output format which can takes those values: "date"
            or "decimal"
        ----
        OUTPUT:
            @return: str - Local Mean Solar Time

        """
        if utc_date is None:
            utc_date = UTCDateTime().now()

        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)

        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)

        origin_in_sec = self.__origindate.timestamp
        any_date_in_sec = utc_date.timestamp

        delta_sec = any_date_in_sec - origin_in_sec
        raw_martian_sol = delta_sec / (self.SECONDS_PER_MARS_DAY)

        nb_sol = int(math.modf(raw_martian_sol)[1])

        hour_dec = 24 * math.modf(raw_martian_sol)[0]
        ihour = floor(hour_dec)
        # MINUTES
        min_dec = 60*(hour_dec-ihour)
        iminutes = floor(min_dec)
        seconds = (min_dec - iminutes)*60.
        iseconds = int(math.modf((seconds))[1])
        milliseconds = int(math.modf((seconds-iseconds))[0]*1000000)

        if output == "decimal":
            marsDate = raw_martian_sol
        else:
            marsDate = "{:04}T{:02}:{:02}:{:02}.{:06}".format(nb_sol, ihour,
                                                              iminutes,
                                                              iseconds,
                                                              milliseconds)
        return marsDate

    def get_utc_2_lmst_2tab(self, utc_date=None):
        """
        Convert UTC date to LMST date into a list
        ----
        INPUT:
            @utc_date: UTCDateTime
        ----
        OUTPUT:
            @return: list - Local Mean Solar Time
                    [SOL, Hour, Minutes, Second, milliseconds]]
        """
        marsDateTab = []
        marsdate = self.get_utc_2_lmst(utc_date=utc_date, output="date")
        whereTpos = marsdate.find("T")

        if whereTpos > 0:
            # extract the number of SOLS

            nbsol = int(marsdate[:whereTpos])
            marsDateTab.append(nbsol)

            # extract hour time in a list
            timepart = marsdate[whereTpos+1:].split(":")

            if len(timepart) == 2:  # only hh:mm
                marsDateTab.append(int(timepart[0]))
                marsDateTab.append(int(timepart[1]))
            elif len(timepart) == 3:  # hh:mm:ss.sssssss
                marsDateTab.append(int(timepart[0]))
                marsDateTab.append(int(timepart[1]))
                marsDateTab.append(float(timepart[2]))

            else:
                marsDateTab.append(int(timepart[0]))
            return marsDateTab
        else:
            return None

    def get_lmst_to_utc(self, lmst_date=None):
        """
        Function to estimate the UTC time giving a LMST time.
        LMST Time must have the following formar : XXXXTMM:MM:ss.mmm
        with :
            SSSS : number of sols
            HH: Hours
            MM: Minutes
            ss: Seconds
            mmm: miliseconds
        ----
        INPUT
            @lmst_date: string
        ----
        OUPUT
            @return: Time with UTCDateTime format
        """

        if lmst_date is None:
            return UTCDateTime.now()
        else:
            date2split = str(lmst_date)
            whereTpos = date2split.find("T")
            if whereTpos > 0:
                # extract the number of SOLS
                nbsol = float(date2split[:whereTpos])

                # extract hour time in a list
                timepart = date2split[whereTpos+1:].split(":")

                #print(f"value of timepart: {timepart}")

                if len(timepart) == 2:  # only hh:mm
                    hours_in_dec = float(timepart[0]) + float(timepart[1])/60
                    #print(f"Value of hours in dec (n=2): {hours_in_dec}")
                elif len(timepart) == 3:  # hh:mm:ss.sssssss
                    hours_in_dec = float(timepart[0])+float(timepart[1])/60 + \
                                float(timepart[2])/(60*60)
                    #print(f"Value of hours in dec (n=3): {hours_in_dec}")
                else:
                    hours_in_dec = None
                    #print(f"Value of hours in dec (n=other): {hours_in_dec}")

                jd_utc_orig = self.utcDateTime_to_jdutc(self.get_origindate())
                jd_tt_orig = self.julian_tt(jday_utc=jd_utc_orig)

                jd_ott_orig = self.j2000_offset_tt(jd_tt_orig)

                MTC = self.Coordinated_Mars_Time(jd_ott_orig)

                # Add the number of SOL to the MTC of the origin date
                MTC += nbsol*24
                #print(f"in get_lmst_to_utc -> MTC = {MTC}")
                # Add the number of hours to the MTC of the origin date
                if hours_in_dec is not None:
                    MTC += hours_in_dec

                    # Get back to Delta J2000 (Eq 15)
                    JD_OTT = (MTC/24 - self.KNORM + self.K)*self.SOL_RATIO+4.5

                    # Equation A6 from MARS 24
                    # (https://www.giss.nasa.gov/tools/mars24/help/algorithm.html)
                    JD_TT = JD_OTT + self.CONSISTENT_JULIAN_DAY

                    # Equation A2 from MARS 24
                    JD_UT = JD_TT - 69.184/86400

                    # Equation A1 from MARS 24
                    UTC = (JD_UT - self.JULIAN_UNIX_EPOCH)*self.MILLISECONDS_IN_A_DAY/1000.

                    return UTCDateTime(UTC)
                else:
                    return None
            # Case where you just give the a number of SOL
            elif whereTpos < 0 or not whereTpos:

                # Extract the MTC time of the "origin time"
                # (time where SOL 0 starts)

                jd_utc_orig = self.utcDateTime_to_jdutc(self.get_origindate())
                jd_tt_orig = self.julian_tt(jday_utc=jd_utc_orig)

                jd_ott_orig = self.j2000_offset_tt(jd_tt_orig)

                MTC = self.Coordinated_Mars_Time(jd_ott_orig)
                MTC += float(date2split)*24

                # Get back to Delta J2000 (Eq 15)
                JD_OTT = (MTC/24 - self.KNORM + self.K)*self.SOL_RATIO + 4.5

                # Equation A6 from MARS 24
                # (https://www.giss.nasa.gov/tools/mars24/help/algorithm.html)
                JD_TT = JD_OTT + self.CONSISTENT_JULIAN_DAY

                # Equation A2 from MARS 24
                JD_UT = JD_TT - 69.184/86400

                # Equation A1 from MARS 24
                UTC = (JD_UT - self.JULIAN_UNIX_EPOCH)*self.MILLISECONDS_IN_A_DAY/1000.

                correction_factor = .466
                return UTCDateTime(UTC)+correction_factor
            else:
                return None

    # =======================================================================
    # Additionnal Calculations
    #    added in 19', Nov 26th
    # =======================================================================
    def solar_declination(self, utc_date=None):
        """
        Determine solar declination (planetographic). (AM1997, eq. D5)
        ----
        INPUT:
            @utc_date:

        """
        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)

        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)

        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)
        ls = self.L_s(jd_ott)
        delta_s = (180/math.pi)*math.asin(0.42565*math.sin(math.pi*ls/180)) \
            + 0.25 * math.sin(math.pi*ls/180)   # (-> AM1997, eq. D5)
        return delta_s

    def local_solar_elevation(self, utc_date=None):
        """
        For any given point on Mars's surface,
        we want to determine the angle of the sun.
        From section D-5 on Mars24 algo page
        added in dec 19, 19th
        """
        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)

        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)
        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)

        MTC = self.Coordinated_Mars_Time(j2000_ott=jd_ott)
        MTC = MTC % 24
        delta_s = self.solar_declination(utc_date=utc_date)

        lbda = self.LONGITUDE
        lbda = lbda % 360
        phi = self.LATITUDE

        EOT = self.equation_of_time(j2000_ott=jd_ott)
        lbda_s = MTC * (360/24) + EOT + 180
        lbda_s = lbda_s % 360
        d2r = math.pi / 180.
        H = lbda - lbda_s
        Z = (180/math.pi) * math.acos(math.sin(delta_s * d2r)\
                                      * math.sin(phi*d2r)\
                                          + math.cos(delta_s*d2r)\
                                          * math.cos(phi*d2r)\
                                              * math.cos(H * d2r))
        solar_elevation = 90 - Z
        return solar_elevation

    def local_solar_azimuth(self, utc_date=None):
        """
        For any given point on Mars's surface,
        we want to determine the angle of the sun.
        From section D-6 on Mars24 algo page
        added in dec 19, 19th
        """
        if isinstance(utc_date, UTCDateTime):
            jd_utc = self.utcDateTime_to_jdutc(utc_date)

        elif isinstance(utc_date, str):
            try:
                utc_date_in_utc = UTCDateTime(utc_date)
                utc_date = utc_date_in_utc
            except TypeError:
                return None
            else:
                jd_utc = self.utcDateTime_to_jdutc(utc_date_in_utc)
        jd_tt = self.julian_tt(jday_utc=jd_utc)
        jd_ott = self.j2000_offset_tt(jd_tt)
        MTC = self.Coordinated_Mars_Time(j2000_ott=jd_ott)
        MTC = MTC % 24
        # print("\t -MTC= ", MTC)
        delta_s = self.solar_declination(utc_date=utc_date)
        # print("\t -deltas= ", delta_s)

        lbda = self.LONGITUDE
        lbda = lbda % 360
        phi = self.LATITUDE

        EOT = self.equation_of_time(j2000_ott=jd_ott)
        lbda_s = MTC*(360/24) + EOT + 180
        lbda_s = lbda_s % 360
        # print("\t -lbda_s=", lbda_s)
        d2r = math.pi/180
        H = lbda - lbda_s

        A = (180/math.pi) * math.atan(math.sin(H * d2r)/\
                       (math.cos(phi * d2r) * math.tan(delta_s * d2r)\
                        - math.sin(phi * d2r) * math.cos(H * d2r)))
        A = A % 360
        return A

#%%

#WindSightNet data
FullWind = pd.read_csv('WindSightNet.csv')  

# TWINS data
OrigWind = pd.read_csv('TWINS.csv')

#%%




Solnums = np.arange(1440)+60



Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]



#%%
list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)

             

#%% Assign to bins 

FullWind["LTST_hr"] = 40
OrigWind["LTST_hr"] = 40

conditions2 = [
    (OrigWind['LTST'] < 2),
    (OrigWind['LTST'] >= 2) & (OrigWind['LTST'] < 6),
    (OrigWind['LTST'] >= 6) & (OrigWind['LTST'] < 10),
    (OrigWind['LTST'] >= 10) & (OrigWind['LTST'] < 14),
    (OrigWind['LTST'] >= 14) & (OrigWind['LTST'] < 18),
    (OrigWind['LTST'] >= 18) & (OrigWind['LTST'] < 22),
    (OrigWind['LTST'] >= 22)
]

conditions = [
    (FullWind['LTST'] < 2),
    (FullWind['LTST'] >= 2) & (FullWind['LTST'] < 6),
    (FullWind['LTST'] >= 6) & (FullWind['LTST'] < 10),
    (FullWind['LTST'] >= 10) & (FullWind['LTST'] < 14),
    (FullWind['LTST'] >= 14) & (FullWind['LTST'] < 18),
    (FullWind['LTST'] >= 18) & (FullWind['LTST'] < 22),
    (FullWind['LTST'] >= 22)
]

values = ['22-02', '02-06', '06-10', '10-14' , '14-18', '18-22', '22-02']

FullWind['LTST_hr'] = np.select(conditions, values)

OrigWind['LTST_hr'] = np.select(conditions2, values)


conditions_year = [
    (FullWind['Sol'] < 114),
    (FullWind['Sol'] >=114) & (FullWind['Sol']< 782) ,
    (FullWind['Sol'] > 782) 
]
values = ["MY 34", "MY 35", "MY 36"]
FullWind['Mars_Year'] = np.select(conditions_year, values)

conditions_yearorig = [
    (OrigWind['Sol'] < 114),
    (OrigWind['Sol'] >=114 )& (OrigWind['Sol']< 782) ,
    (OrigWind['Sol'] > 782) 
]

values = ["TWINS MY 34", "TWINS MY 35", "TWINS MY 36"]
OrigWind['Mars_Year'] = np.select(conditions_yearorig, values)
   
         

FullWind["L_s_bin"] = np.ceil((FullWind["L_s"]/60))   

OrigWind["L_s_bin"] = np.ceil((OrigWind["L_s"]/60))  

conditions_ls = [
    (FullWind['L_s_bin'] == 1),
    (FullWind['L_s_bin'] == 2) ,
    (FullWind['L_s_bin'] == 3) ,
    (FullWind['L_s_bin'] == 4) ,
    (FullWind['L_s_bin'] == 5) ,
    (FullWind['L_s_bin'] == 6) 
]

conditions_ls2 = [
    (OrigWind['L_s_bin'] == 1),
    (OrigWind['L_s_bin'] == 2) ,
    (OrigWind['L_s_bin'] == 3) ,
    (OrigWind['L_s_bin'] == 4) ,
    (OrigWind['L_s_bin'] == 5) ,
    (OrigWind['L_s_bin'] == 6) 
]
values = ["0-60", "60-120", "120-180", "180-240" , "240-300", "300-360"]


FullWind['L_s_bin'] = np.select(conditions_ls, values)

OrigWind['L_s_bin'] = np.select(conditions_ls2, values)

#%% Calculate Ls and sols
Solnums = np.arange(1440)+60
mDate = MarsConverter()

SolMid = np.zeros(1440)
UTCmid = []
idx = 0
for sol in Solnums:
#    if sol <100:
    
    date_string = str(sol)+"T12:00:00"
    UTCDate = mDate.get_lmst_to_utc(lmst_date=date_string)
    marsDate = mDate.get_utc_2_ls(utc_date=UTCDate) 
    
    SolMid[idx] =marsDate
    idx=idx+1
    
LsVec = np.arange(0, 360, 60)
sols_Ls = np.zeros((len(LsVec)*5,2))
Lsvecin = FullWind["L_s"]
idx=1
for Ls in LsVec:
    print(Ls)
    for sol in np.arange(1440):
        if np.abs(SolMid[sol]-Ls) < 0.5:
            if np.abs(sols_Ls[idx-1,1]-sol)>15:
                sols_Ls[idx,0] = Ls
                sols_Ls[idx,1] = sol+60
                idx=idx+1
                
Ls_sol_save = sols_Ls[1:23,:]
SolsSavetest= Ls_sol_save[:,1]
Ls_sol_save = sols_Ls[1:22,:]
Ls_sol_save = Ls_sol_save[np.abs(np.diff(SolsSavetest))>3,:]


#%% Show data contents - Figure 3 prep

TimeVecBins = np.arange(0,24,100/3600)
Matrix_origwindspeed = 100*np.ones((len(TimeVecBins),1440))
Matrix_origwinddirection = 400*np.ones((len(TimeVecBins),1440))
Matrix_fullwindspeed = 100*np.ones((len(TimeVecBins),1440))
Matrix_fullwinddirection = 400*np.ones((len(TimeVecBins),1440))

templist = [332,333]

for sol in list_sols:
    print(sol)
    DataSel = OrigWind[OrigWind["Sol"]==int(sol)]
    LTSTvec = DataSel["LTST"].to_numpy()
    WindSvec = DataSel["Wind Speed"].to_numpy()
    WindDvec = DataSel["Wind dir."].to_numpy()
    if len(LTSTvec) > 1:
        counter = 0
        for time_ind in TimeVecBins:
            idx = (np.abs(LTSTvec - time_ind)).argmin()
            if (LTSTvec[idx]-time_ind)<150/3600:
                Matrix_origwindspeed[counter,int(sol)] = WindSvec[idx]
                Matrix_origwinddirection[counter,int(sol)] = WindDvec[idx]
            counter = counter+1
    DataSel = FullWind[FullWind["Sol"]==int(sol)]
    LTSTvec = DataSel["LTST"].to_numpy()
    WindSvec = DataSel["Wind Speed"].to_numpy()
    WindDvec = DataSel["Wind dir."].to_numpy()
    if len(LTSTvec) > 1:
        counter = 0
        for time_ind in TimeVecBins:
            idx = (np.abs(LTSTvec - time_ind)).argmin()
            if (LTSTvec[idx]-time_ind)<150/3600:
                Matrix_fullwindspeed[counter,int(sol)] = WindSvec[idx]
                Matrix_fullwinddirection[counter,int(sol)] = WindDvec[idx]
            counter = counter+1
            
            
    
Solnums = np.arange(1440)+60


SolMid = np.zeros(1440)
UTCmid = []
idx = 0
for sol in Solnums:
#    if sol <100:
    
    date_string = str(sol)+"T12:00:00"
    UTCDate = mDate.get_lmst_to_utc(lmst_date=date_string)
    marsDate = mDate.get_utc_2_ls(utc_date=UTCDate) 
    
    SolMid[idx] =marsDate
    idx=idx+1


LsVec = np.arange(0, 360, 60)
sols_Ls = np.zeros((len(LsVec)*5,2))
Lsvecin = FullWind["L_s"]
idx=1
for Ls in LsVec:
    print(Ls)
    for sol in np.arange(1440):
        if np.abs(SolMid[sol]-Ls) < 0.5:
            if np.abs(sols_Ls[idx-1,1]-sol)>15:
                sols_Ls[idx,0] = Ls
                sols_Ls[idx,1] = sol+60
                idx=idx+1
   
Ls_sol_save = sols_Ls[1:23,:]

SolsSavetest= Ls_sol_save[:,1]
Ls_sol_save = sols_Ls[1:22,:]
Ls_sol_save = Ls_sol_save[np.abs(np.diff(SolsSavetest))>3,:]


#%% Figure 3 plotting
X, Y = np.meshgrid(np.arange(1441),np.arange(0,24+100/36000,100/3600))

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


levels = MaxNLocator(nbins=60).tick_values(0, 15)

cmap = plt.colormaps['bone']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


fig, ax = plt.subplots(4, 1, figsize=(15,20))

im = ax[0].pcolormesh(X,Y,Matrix_origwindspeed, cmap=cmap, norm=norm)
axins = ax[0].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('Wind speed', size = 18)
cbar.ax.tick_params(labelsize=14) 

ax[0].set_title('TWINS wind speed', size = 20)
ax[0].set_xlabel("Sol", size = 18)
ax[0].set_ylabel("LTST hr", size = 18)
ax2 = ax[0].twiny()
ax2.set_xticks(Ls_sol_save[:,1])
ax2.set_xticklabels(Ls_sol_save[:,0])
ax2.set_xlabel("Ls")


im = ax[1].pcolormesh(X,Y,Matrix_fullwindspeed, cmap=cmap, norm=norm)
ax[1].set_title('Seismic wind speed', size = 20)
ax[1].set_xlabel("Sol", size = 18)
ax[1].set_ylabel("LTST hr", size = 18)
axins = ax[1].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('Wind speed', size = 18)
cbar.ax.tick_params(labelsize=14) 

ax2 = ax[1].twiny()
ax2.set_xticks(Ls_sol_save[:,1])
ax2.set_xticklabels(Ls_sol_save[:,0])
ax2.set_xlabel("Ls")


fig.tight_layout()


levels = MaxNLocator(nbins=60).tick_values(0, 365)

cmap = plt.colormaps['twilight']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


im = ax[2].pcolormesh(X,Y,Matrix_origwinddirection, cmap=cmap, norm=norm)
ax[2].set_xlabel("Sol", size = 18)
ax[2].set_title('TWINS wind direction', size = 20)
ax[2].set_ylabel("LTST hr", size = 18)
#fig.colorbar(im, ax=ax[2])
axins = ax[2].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('Wind direction', size = 18)
cbar.ax.tick_params(labelsize=14) 

ax2 = ax[2].twiny()
ax2.set_xticks(Ls_sol_save[:,1])
ax2.set_xticklabels(Ls_sol_save[:,0])
ax2.set_xlabel("Ls")


#fig = plt.figure(figsize=(10,4))
im = ax[3].pcolormesh(X,Y,Matrix_fullwinddirection, cmap=cmap, norm=norm)
ax[3].set_xlabel("Sol", size = 18)
ax[3].set_title('Seismic wind direction', size = 20)

ax[3].set_ylabel("LTST hr", size = 18)
#fig.colorbar(im, ax=ax[3])
axins = ax[3].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('Wind direction', size = 18)
cbar.ax.tick_params(labelsize=14) 


ax2 = ax[3].twiny()
ax2.set_xticks(Ls_sol_save[:,1])
ax2.set_xticklabels(Ls_sol_save[:,0])
ax2.set_xlabel("Ls", size = 18)

ax[0].tick_params(axis='both', labelsize=14)
ax[1].tick_params(axis='both', labelsize=14)
ax[2].tick_params(axis='both', labelsize=14)
ax[3].tick_params(axis='both', labelsize=14)

fig.tight_layout()



#%% Example sols - Figure 5
Solnums = np.arange(1400)+60


Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]

print(Input_Sols)

list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)

Percentiles_origwind = 100*np.ones((1400,5))
Percentiles_fullwind = 100*np.ones((1400,5))

fig, ax = plt.subplots(6, 2, figsize=(15,15))
for sol in list_sols:
    print(sol)
    DataSel = FullWind[FullWind["Sol"]==int(sol)]
    WindSvec = DataSel["Wind Speed"].to_numpy()
    WindDvec = DataSel["Wind dir."].to_numpy()
    LTST = DataSel["LTST"].to_numpy()
    if len(WindSvec)>500:
        if (DataSel["L_s_bin"].iloc[0]=="0-60"):
            ax[0,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[0,1].plot(LTST,WindDvec,color='silver',alpha=0.1)
        if (DataSel["L_s_bin"].iloc[0]=="60-120"):
            ax[1,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[1,1].plot(LTST,WindDvec,color='silver',alpha=0.1)
        if (DataSel["L_s_bin"].iloc[0]=="120-180"):
            ax[2,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[2,1].plot(LTST,WindDvec,color='silver',alpha=0.1)
        if (DataSel["L_s_bin"].iloc[0]=="180-240"):
            ax[3,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[3,1].plot(LTST,WindDvec,color='silver',alpha=0.1)
        if (DataSel["L_s_bin"].iloc[0]=="240-300"):
            ax[4,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[4,1].plot(LTST,WindDvec,color='silver',alpha=0.1)
        if (DataSel["L_s_bin"].iloc[0]=="300-360"):
            ax[5,0].plot(LTST,WindSvec,color='silver',alpha=0.1)
            ax[5,1].plot(LTST,WindDvec,color='silver',alpha=0.1)

DataSel = FullWind[FullWind["Sol"]==int(220+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[0,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(220+668))
ax[0,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(220+668))

DataSel = FullWind[FullWind["Sol"]==int(220)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[0,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(220))
ax[0,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(220))


DataSel = OrigWind[OrigWind["Sol"]==int(220)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[0,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(220))
ax[0,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(220))


ax[0,0].set_ylabel('Wind speed m/s',size=14)
ax[0,1].set_ylabel('Wind dir. deg',size=14)
ax[1,0].set_ylabel('Wind speed m/s',size=14)
ax[1,1].set_ylabel('Wind dir. deg',size=14)
ax[2,0].set_ylabel('Wind speed m/s',size=14)
ax[2,1].set_ylabel('Wind dir. deg',size=14)
ax[3,0].set_ylabel('Wind speed m/s',size=14)
ax[3,1].set_ylabel('Wind dir. deg',size=14)
ax[4,0].set_ylabel('Wind speed m/s',size=14)
ax[4,1].set_ylabel('Wind dir. deg',size=14)
ax[5,0].set_ylabel('Wind speed m/s',size=14)
ax[5,1].set_ylabel('Wind dir. deg',size=14)

ax[0,0].set_xlabel('LTST hr',size=14)
ax[0,1].set_xlabel('LTST hr',size=14)
ax[1,0].set_xlabel('LTST hr',size=14)
ax[1,1].set_xlabel('LTST hr',size=14)
ax[2,0].set_xlabel('LTST hr',size=14)
ax[2,1].set_xlabel('LTST hr',size=14)
ax[3,0].set_xlabel('LTST hr',size=14)
ax[3,1].set_xlabel('LTST hr',size=14)
ax[4,0].set_xlabel('LTST hr',size=14)
ax[4,1].set_xlabel('LTST hr',size=14)
ax[5,0].set_xlabel('LTST hr',size=14)
ax[5,1].set_xlabel('LTST hr',size=14)

ax[0,0].set_title('L_s = 0-60',size=14)
ax[0,1].set_title('L_s = 0-60',size=14)
ax[1,0].set_title('L_s = 60-120',size=14)
ax[1,1].set_title('L_s = 60-120',size=14)
ax[2,0].set_title('L_s = 120-180',size=14)
ax[2,1].set_title('L_s = 120-180',size=14)
ax[3,0].set_title('L_s = 180-240',size=14)
ax[3,1].set_title('L_s = 180-240',size=14)
ax[4,0].set_title('L_s = 240-300',size=14)
ax[4,1].set_title('L_s = 240-300',size=14)
ax[5,0].set_title('L_s = 300-360',size=14)
ax[5,1].set_title('L_s = 300-360',size=14)

DataSel = FullWind[FullWind["Sol"]==int(333+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[1,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(333+668))
ax[1,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(333+668))

DataSel = FullWind[FullWind["Sol"]==int(333)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[1,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(333))
ax[1,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(333))


DataSel = OrigWind[OrigWind["Sol"]==int(333)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[1,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(333))
ax[1,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(333))


DataSel = FullWind[FullWind["Sol"]==int(414+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[2,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(414+668))
ax[2,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(414+668))

DataSel = FullWind[FullWind["Sol"]==int(414)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[2,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(414))
ax[2,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(414))


DataSel = OrigWind[OrigWind["Sol"]==int(414)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[2,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(414))
ax[2,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(414))


DataSel = FullWind[FullWind["Sol"]==int(500+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[3,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(500+668))
ax[3,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(500+668))

DataSel = FullWind[FullWind["Sol"]==int(500)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[3,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(500))
ax[3,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(500))


DataSel = OrigWind[OrigWind["Sol"]==int(500)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[3,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(500))
ax[3,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(500))


DataSel = FullWind[FullWind["Sol"]==int(590+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[4,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(590+668))
ax[4,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(590+668))

DataSel = FullWind[FullWind["Sol"]==int(590)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[4,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(590))
ax[4,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(590))


DataSel = OrigWind[OrigWind["Sol"]==int(590)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[4,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(590))
ax[4,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(590))

DataSel = FullWind[FullWind["Sol"]==int(690+668)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[5,0].plot(LTST,WindSvec,color='indianred',alpha=0.4, label="Sol "+str(690+668))
ax[5,1].plot(LTST,WindDvec,color='indianred',alpha=0.4, label="Sol "+str(690+668))

DataSel = FullWind[FullWind["Sol"]==int(690)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[5,0].plot(LTST,WindSvec,color='maroon',alpha=0.8, label="Sol "+str(690))
ax[5,1].plot(LTST,WindDvec,color='maroon',alpha=0.8, label="Sol "+str(690))


DataSel = OrigWind[OrigWind["Sol"]==int(690)]
WindSvec = DataSel["Wind Speed"].to_numpy()
WindDvec = DataSel["Wind dir."].to_numpy()
LTST = DataSel["LTST"].to_numpy()
ax[5,0].plot(LTST,WindSvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(690))
ax[5,1].plot(LTST,WindDvec,color='royalblue',alpha=0.8, label="TWINS Sol "+str(690))


ax[0,0].set_xlim((0,24))
ax[0,0].set_ylim((0,20))
ax[0,1].set_xlim((0,24))
ax[0,1].set_ylim((0,360))
ax[2,0].set_xlim((0,24))
ax[2,0].set_ylim((0,20))
ax[2,1].set_xlim((0,24))
ax[2,1].set_ylim((0,360))
ax[3,0].set_xlim((0,24))
ax[3,0].set_ylim((0,20))
ax[3,1].set_xlim((0,24))
ax[3,1].set_ylim((0,360))
ax[4,0].set_xlim((0,24))
ax[4,0].set_ylim((0,20))
ax[4,1].set_xlim((0,24))
ax[4,1].set_ylim((0,360))
ax[5,0].set_xlim((0,24))
ax[5,0].set_ylim((0,20))
ax[5,1].set_xlim((0,24))
ax[5,1].set_ylim((0,360))
ax[1,0].set_xlim((0,24))
ax[1,0].set_ylim((0,20))
ax[1,1].set_xlim((0,24))
ax[1,1].set_ylim((0,360))

ax[0,0].legend(fontsize=12,ncol=3)
ax[0,1].legend(fontsize=12,ncol=3)
ax[1,0].legend(fontsize=12,ncol=3)
ax[1,1].legend(fontsize=12,ncol=3)
ax[2,0].legend(fontsize=12,ncol=3)
ax[2,1].legend(fontsize=12,ncol=3)
ax[3,0].legend(fontsize=12,ncol=3)
ax[3,1].legend(fontsize=12,ncol=3)
ax[4,0].legend(fontsize=12,ncol=3)
ax[4,1].legend(fontsize=12,ncol=3)
ax[5,0].legend(fontsize=12,ncol=3)
ax[5,1].legend(fontsize=12,ncol=3)


fig.tight_layout()

#%% percentile plot - Figure 4

Solnums = np.arange(1400)+0

Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]

print(Input_Sols)


list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)

Percentiles_origwind = 100*np.ones((1400,5))
Percentiles_fullwind = 100*np.ones((1400,5))
mean_fullwind = 100*np.ones((1400,1))
Ls_fullwind = 100*np.ones((1400,1))

for sol in list_sols:
    print(sol)
    DataSel = OrigWind[OrigWind["Sol"]==int(sol)]
    WindSvec = DataSel["Wind Speed"].to_numpy()
    WindSvec = WindSvec[WindSvec>=0]
    WindDvec = DataSel["Wind dir."].to_numpy()
    #WindDvec = WindDvec[WindSvec>=0]
    if len(WindSvec) > 600:
        Percentiles_origwind[int(sol),:] = np.percentile(WindSvec, [0, 25, 50, 75, 100])
    else:
        Percentiles_origwind[int(sol),:] = [np.nan,np.nan,np.nan,np.nan,np.nan]
    DataSel = FullWind[FullWind["Sol"]==int(sol)]
    WindSvec = DataSel["Wind Speed"].to_numpy()
    WindDvec = DataSel["Wind dir."].to_numpy()
    if len(WindSvec) > 600:
        Percentiles_fullwind[int(sol),:] = np.percentile(WindSvec, [0, 25, 50, 75, 100])
        mean_fullwind[int(sol)] = np.mean(WindSvec)
    else:
        Percentiles_fullwind[int(sol),:] = [np.nan,np.nan,np.nan,np.nan,np.nan]
        mean_fullwind[int(sol)] = np.nan


fig, ax = plt.subplots(1, 1, figsize=(12,6))

ax.plot(np.arange(0,1400),Percentiles_fullwind[:,0],color='palegoldenrod',label='0th seis')
ax.plot(np.arange(0,1400),Percentiles_fullwind[:,1],color='darkseagreen',label='25th seis')
ax.plot(np.arange(0,1400),Percentiles_fullwind[:,2],color='deepskyblue',label='50th seis')
ax.plot(np.arange(0,1400),Percentiles_fullwind[:,3],color='thistle',label='75th seis')
ax.plot(np.arange(0,1400),Percentiles_fullwind[:,4],color='indianred',label='100th seis')

ax.plot(np.arange(0,1400),Percentiles_origwind[:,0],'--',color='gold',label='0th TWINS')
ax.plot(np.arange(0,1400),Percentiles_origwind[:,1],'--',color='darkolivegreen',label='25th TWINS')
ax.plot(np.arange(0,1400),Percentiles_origwind[:,2],'--',color='midnightblue',label='50th TWINS')

ax.plot(np.arange(0,1400),Percentiles_origwind[:,3],'--',color='indigo',label='75th TWINS')
ax.plot(np.arange(0,1400),Percentiles_origwind[:,4],'--',color='maroon',label='100th TWINS')
ax.grid()
ax.set_xlabel("Sol", size = 16)
ax.set_ylabel("Wind Speed m/s", size = 16)
ax.set_ylim((-1,25))
ax.legend(fontsize=14,ncol=4)
ax.tick_params(axis='both', labelsize=14)
ax2 = ax.twiny()
ax2.set_xticks(Ls_sol_save[:,1])
ax2.set_xticklabels(Ls_sol_save[:,0],size=14)
ax2.set_xlabel("Ls", size = 16)

#%%
AllData = pd.concat([OrigWind[OrigWind["Mars_Year"]=="TWINS MY 35"],FullWind[FullWind["Mars_Year"].isin(['MY 35','MY 36'])]])


#%% Rose plot - Figure 6


from matplotlib.gridspec import GridSpec

values_LT = ['22-02', '02-06', '06-10', '10-14', '14-18', '18-22']

values_Ls = ["0-60", "60-120", "120-180", "180-240" , "240-300", "300-360"]

Ls_bins = ["0-60", "60-120", "120-180", "180-240" , "240-300", "300-360"]


fig = plt.figure(figsize=(12,12))
gs = GridSpec(nrows=6, ncols=6)

row = 0
col = 0


for Ls in values_Ls:

    for LT in values_LT: 

        DataSel = AllData[AllData["Mars_Year"].isin(["TWINS MY 35"])]
        DataSel = DataSel[DataSel["LTST_hr"]==LT]
        DataSel = DataSel[DataSel["L_s_bin"]==Ls]
        a = DataSel["Wind dir."].to_numpy()

        hist_twins, bin_edges_twins = np.histogram(a, density=False, bins=50,range=(0,360))

        DataSel = AllData[AllData["Mars_Year"].isin(['MY 35'])]
        DataSel = DataSel[DataSel["LTST_hr"]==LT]
        DataSel = DataSel[DataSel["L_s_bin"]==Ls]
        a = DataSel["Wind dir."].to_numpy()
        hist_year1, bin_edges_year1 = np.histogram(a, density=False, bins=50,range=(0,360))

        DataSel = AllData[AllData["Mars_Year"].isin(['MY 36'])]
        DataSel = DataSel[DataSel["LTST_hr"]==LT]
        DataSel = DataSel[DataSel["L_s_bin"]==Ls]
        a = DataSel["Wind dir."].to_numpy()
        hist_year2, bin_edges_year2 = np.histogram(a, density=False, bins=50,range=(0,360))

        hist_twins = hist_twins/max(hist_twins)
        hist_year2 = hist_year2/max(hist_year2)
        hist_year1 = hist_year1/max(hist_year1)
        ax = fig.add_subplot(gs[col, row], projection='polar')
        ax.set_theta_offset(np.pi / 2)

        bined = bin_edges_year1[:-1] * (2*np.pi) /360
        ax.set_theta_direction(-1)
            
        ax.set_rlabel_position(0)
        
            
        ax.plot(bined, hist_twins,color="slategray", linewidth=2.5, linestyle='solid',alpha=0.8, label='TWINS MY 35')
        ax.plot(bined, hist_year1,color="indianred", linewidth=2.5, linestyle='solid',alpha=0.8, label='MY 35')
        ax.plot(bined, hist_year2,color="darkgreen", linewidth=2.5, linestyle='solid',alpha=0.8, label='MY 36')
        
        #ax.legend()

        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        
        #ax.set_axis_off()

        col = col+1
    col = 0
    row = row+1
fig.tight_layout()

#%% Wind speed distribution plot - Figure 7

AllData = AllData[AllData['Sol']>114]

Ls_bins = ["0-60", "60-120", "120-180", "180-240" , "240-300", "300-360"]
values_LT = ['22-02', '02-06', '06-10', '10-14', '14-18', '18-22']

Solnums = np.arange(1400)+60



Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]

print(Input_Sols)


list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)

fig, ax = plt.subplots(6, 1, figsize=(8,15))

col = 0
row = 0
for LT in values_LT:

    DataSel = AllData[AllData["LTST_hr"]==LT]
    DataSel = DataSel[DataSel["Mars_Year"].isin(['MY 35','MY 36'])]
    
    
    sns.violinplot(data=DataSel, x = "L_s_bin", y="Wind Speed",  hue="Mars_Year",
                   split=True, inner="quart", fill=False, ax=ax[col], palette="Paired")
    
    sns.move_legend(
        ax[col], "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title="LTST range = "+LT,  frameon=False, title_fontsize = 14
        )
    
    ax[col].set_ylabel("Wind speed m/s",size = 12)
    ax[col].set_xlabel("L_s deg",size = 12)

    ax[col].tick_params(axis='both', labelsize=12)
    col=col+1

fig.tight_layout()

#%% Turbulent cells - Figure 8
from datetime import datetime
import matplotlib
from matplotlib.mlab import psd


Solnums = np.arange(1400)+60

Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]

print(Input_Sols)


list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)



ConstVec= np.arange(14,16,100/3600)

storeFreq = np.zeros((1500,(int(len(ConstVec)/2 +1))))

storePSD =6000* np.ones((1500,(int(len(ConstVec)/2 +1))))


storeWind14 = np.zeros(1500)


for sol in list_sols:
    
    DataSel = FullWind[FullWind["Sol"]==int(sol)]
    #DataSel = OrigWind[OrigWind["Sol"]==int(sol)]
    
    
    DataSel = DataSel[DataSel["LTST"]>14] 
    DataSel = DataSel[DataSel["LTST"]<16]
    WindSvec = DataSel["Wind Speed"].to_numpy()
    Timevec = DataSel["LTST"].to_numpy()

    if len(Timevec) >50:

        storeWind14[int(sol)] = np.nanmean(WindSvec)

        ConstVec= np.arange(np.min(Timevec),np.max(Timevec),100/3600)

        InterWind = sp.interpolate.interp1d(Timevec, WindSvec, kind='nearest')

        WindFull = InterWind(ConstVec)
    
        Pxx, freq = psd(sp.signal.detrend(WindFull-np.mean(WindFull)), NFFT=len(ConstVec) // 1 , pad_to=len(ConstVec), noverlap=int(0.1 * len(ConstVec)), Fs=1/100)
    
        storePSD[int(sol),0:len(Pxx)] = (Pxx)/max((Pxx))

        storeFreq[int(sol),0:len(Pxx)] = freq
        
        
        
#%% Plot Figure 8
sol=333
storeFreq[int(sol),:]

storePSD[storePSD>50] = np.nan 

filtlen = 3
PSD_smooth = sp.ndimage.median_filter(storePSD,size=filtlen, axes=0)
PSD_smooth =  sp.ndimage.convolve1d(PSD_smooth, np.ones(filtlen)/filtlen, axis=0, mode='nearest')  




X, Y = np.meshgrid(np.concatenate((storeFreq[int(sol),:].reshape(-1,), storeFreq[int(sol),-1].reshape(-1,)),axis=0),np.arange(1501))



from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


levels = MaxNLocator(nbins=40).tick_values(0, 0.7)

cmap = plt.colormaps['seismic']
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


fig, ax = plt.subplots(1, 1, figsize=(5,15))

im = ax.pcolormesh(X,Y,PSD_smooth, cmap=cmap, norm=norm)

axins = ax.inset_axes([1.3, 0, 0.025,1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('PSD Normalised dB', size = 18)
cbar.ax.tick_params(labelsize=14) 
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlim((storeFreq[int(sol),1],0.004))
ax.set_title('PSD 14-16 LTST', size = 20)
ax.set_ylabel("Sol", size = 18)
ax.set_xlabel("Freq Hz", size = 18)
ax2 = ax.twinx()
ax2.set_yticks(Ls_sol_save[:,1])
ax2.set_yticklabels(Ls_sol_save[:,0], size = 12)
ax2.set_ylabel("Ls", size = 18)




#%% Interpolate Continuous data for Figure 9

allwindvec = FullWind['Wind Speed'].to_numpy()

time_vec = FullWind['Time'].to_numpy()

InterWind = sp.interpolate.interp1d(time_vec, allwindvec, kind='nearest', bounds_error=False,fill_value="extrapolate")

sr = 0.01

start = time_vec[0]
end = time_vec[-1]

increment = time_vec[4]-time_vec[3]

ConstTimeVect = np.arange(start,end,increment)


allwindvec_orig = OrigWind['Wind Speed'].to_numpy()
time_vec_orig = OrigWind['Time'].to_numpy()

InterWindOrig = sp.interpolate.interp1d(time_vec_orig, allwindvec_orig, kind='nearest', bounds_error=False,fill_value="extrapolate")


#%%
landerconfigfile = './marsconverter-master/landerconfig.xml'
my_file = Path(landerconfigfile)

mDate = MarsConverter()

LTSTvec = np.zeros(len(ConstTimeVect))
Solvec = np.zeros(len(ConstTimeVect))
LSvec = np.zeros(len(ConstTimeVect))

UTC_tot = np.empty(len(ConstTimeVect), dtype = UTCDateTime)
for i in range(len(ConstTimeVect)):
    UTC_tot[i] = UTCDateTime(ConstTimeVect[i] * 3600*24 )
    UTCDate = UTC_tot[i]
    marsDate = mDate.get_utc_2_ltst(utc_date=UTCDate, output="decimal")
    #posT = marsDate.find('T')
    LTSTvec[i] = (marsDate)#[posT+1:posT+9])
    marsDate = mDate.get_utc_2_ltst(utc_date=UTCDate, output="date")
    posT = marsDate.find('T')
    Solvec[i] = (marsDate[:posT])
    UTCDate = UTC_tot[i]
    marsDate = mDate.get_utc_2_ls(utc_date=UTCDate)    #posT = marsDate.find('T')
    LSvec[i] = (marsDate)#[posT+1:posT+9])  
#%%
WindFull = InterWind(ConstTimeVect)

WindFull_orig = InterWindOrig(ConstTimeVect)


x_dec = sp.signal.decimate(WindFull, 10)

x_orig_dec = sp.signal.decimate(WindFull_orig, 10)

sr = 0.01/10
Solvec_dec = Solvec[::10]

widths = np.logspace(1.4, 3.8,200)
#%%
cwtmatr, freqs = pywt.cwt(x_dec, widths, 'morl', sampling_period=1/sr)

#%% Plot Figure 9 - WindSightNet
fig, ax = plt.subplots(1, 1, figsize=(8,4))
im = ax.pcolormesh(Solvec_dec, 1/(freqs*(60*60*24.62)), np.log10(np.abs(cwtmatr)), cmap='gist_heat', vmin=0,vmax=2 )

ax.set_yscale('log')
ax.set_ylim(0.9, 40)

ax.tick_params(labelsize=12) 
ax.set_ylabel('Period in Sols',size=14)
ax.set_xlabel('Sol',size=14)
ax.set_title('WindSightNet',size=14)

axins = ax.inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('log(m/s)', size = 12)
cbar.ax.tick_params(labelsize=12) 
#%%
cwtmatr_orig, freqs_orig = pywt.cwt(x_orig_dec, widths, 'morl', sampling_period=1/sr)

#%%Plot Figure 9 - TWINS
fig, ax = plt.subplots(1, 1, figsize=(8,4))
im = ax.pcolormesh(Solvec_dec, 1/(freqs_orig*(60*60*24.62)), np.log10(np.abs(cwtmatr_orig)), cmap='gist_heat', vmin=0,vmax=2 )

ax.set_yscale('log')
ax.set_ylim(0.9, 40)

ax.tick_params(labelsize=12) 
ax.set_ylabel('Period in Sols',size=14)
ax.set_xlabel('Sol',size=14)
ax.set_title('TWINS',size=14)

axins = ax.inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('log(m/s)', size = 12)

cbar.ax.tick_params(labelsize=12) 

#%% Metric plot - Figure 10

interval = 2
interval2 = 20

Ls_start = FullWind[FullWind["Sol"]==114+interval2/2]["L_s"].to_numpy()

Ls_end_year = FullWind[FullWind["Sol"]==668+114+interval2/2]["L_s"].to_numpy()
sr = 0.01


Solnums = np.arange(668)+100

Solnums = Solnums[::interval]


Input_Sols = []
for sol in Solnums:
    if sol <100:
        Input_Sols = Input_Sols + ["SOL"+str(sol).zfill(3)]
    else:
        Input_Sols = Input_Sols + ["SOL"+str(sol)]

print(Input_Sols)


list_sols = []
for sol in Input_Sols:
    solnum = re.findall(r'\d+',sol)
    list_sols += solnum

print(list_sols)
cnt=0
distances = np.zeros(len(Solnums)) 

distances_night = np.zeros(len(Solnums)) 
distances_day = np.zeros(len(Solnums)) 


distancesp = np.zeros(len(Solnums)) 

distances_nightp = np.zeros(len(Solnums)) 
distances_dayp = np.zeros(len(Solnums)) 

Lsbin = np.zeros(len(Solnums)) 

SampDiff = np.zeros(len(Solnums)) 
SampDiff_night = np.zeros(len(Solnums)) 
SampDiff_day = np.zeros(len(Solnums)) 

binage = np.arange(30)/2

WindHist = np.zeros((len(binage)-1,len(Solnums))) 
WindHist2 = np.zeros((len(binage)-1,len(Solnums))) 
Ls_comp = np.zeros(len(Solnums)) 


for sol in list_sols:
    
    DataSel = FullWind[((FullWind["Sol"]>=int(sol)) & (FullWind["Sol"]<(int(sol)+interval2)))]

    WindSvec = DataSel["Wind Speed"].to_numpy()
    
    
    
    DataSel_night = DataSel[~(~(DataSel["LTST"]>=18) & ~(DataSel["LTST"]<=6))]
    DataSel_day = DataSel[(~(DataSel["LTST"]>=18) & ~(DataSel["LTST"]<=6))]
    WindSvec_night = DataSel_night["Wind Speed"].to_numpy()
    WindSvec_day = DataSel_day["Wind Speed"].to_numpy()
    
    
    DataSel_nextyear = FullWind[((FullWind["Sol"]>=(int(sol)+668)) & (FullWind["Sol"]<(int(sol)+interval2+668)))]
    WindSvec_nextyear = DataSel_nextyear["Wind Speed"].to_numpy()
    
    DataSel_nextyear_night = DataSel_nextyear[~(~(DataSel_nextyear["LTST"]>=18) & ~(DataSel_nextyear["LTST"]<=6))]
    DataSel_nextyear_day = DataSel_nextyear[(~(DataSel_nextyear["LTST"]>=18) & ~(DataSel_nextyear["LTST"]<=6))]
    WindSvec_nextyear_night = DataSel_nextyear_night["Wind Speed"].to_numpy()
    WindSvec_nextyear_day = DataSel_nextyear_day["Wind Speed"].to_numpy()
    
    try:
        
        n = np.histogram(WindSvec, binage, density=True)
        WindHist[:,cnt] = n[0].reshape(-1,)
        
        n = np.histogram(WindSvec_nextyear, binage, density=True)
        WindHist2[:,cnt] = n[0].reshape(-1,)
        
        
        Lscontents = DataSel["L_s"].to_numpy()
        Lsmin = Lscontents[0]
        
        Lscontents_nextyear = DataSel_nextyear["L_s"].to_numpy()
        Lsmin_nextyear = Lscontents_nextyear[0]
        


        if np.abs(len(WindSvec)-len(WindSvec_nextyear)) < ((len(WindSvec)+len(WindSvec_nextyear))/1.2):
            
            
            
            distance = sp.stats.wasserstein_distance(WindSvec, WindSvec_nextyear)
            
            distances[cnt] = distance#[0]
            
        
                
            distance = sp.stats.wasserstein_distance(WindSvec_day, WindSvec_nextyear_day)
            distances_day[cnt] = distance
            
        
            distance = sp.stats.wasserstein_distance(WindSvec_night, WindSvec_nextyear_night)
            distances_night[cnt] = distance
        
            Lsbin[cnt] = (DataSel["L_s"].to_numpy()[0]+0.52)#+ DataSel["L_s"].to_numpy()[-1])/2
            SampDiff[cnt] = len(WindSvec) - len(WindSvec_nextyear)
            SampDiff_night[cnt] = len(WindSvec_night) - len(WindSvec_nextyear_night)
            SampDiff_day[cnt] = len(WindSvec_day) - len(WindSvec_nextyear_day)
        
        
        
    except:
        print("Not able to calculate distance for this sol")
    
    cnt=cnt+1
    
    
Solnums2 = np.arange(669)+100
Solnums2 =  Solnums2[::interval]



SolLs = np.zeros(len(Solnums2))
UTCmid = []
idx = 0
for sol in Solnums2:
    date_string = str(sol)+"T12:00:00"
    UTCDate = mDate.get_lmst_to_utc(lmst_date=date_string)
    marsDate = mDate.get_utc_2_ls(utc_date=UTCDate) 
    
    SolLs[idx] =marsDate
    idx=idx+1
    


print(SolLs)


X, Y = np.meshgrid(SolLs,
                   n[1].reshape(-1,))
#%% Plote FIgure 10


Tau_vec = pd.read_csv("insight_tau.csv",skipinitialspace=True)
cmap = plt.colormaps['gist_heat']

levels = MaxNLocator(nbins=30).tick_values(0, 0.4)

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, ax = plt.subplots(4, 1, figsize=(10,18))


im = ax[0].pcolormesh(X,Y,(WindHist), cmap=cmap, norm=norm)
ax[0].set_xlim(0,360)
im = ax[1].pcolormesh(X,Y,(WindHist2), cmap=cmap, norm=norm)
ax[1].set_xlim(0,360)

axins = ax[0].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('PDF density', size = 18)
cbar.ax.tick_params(labelsize=14) 

axins = ax[1].inset_axes([1.025, 0., 0.025, 1])
cbar = fig.colorbar(im, cax=axins, shrink=0.5)
cbar.set_label('PDF density', size = 18)
cbar.ax.tick_params(labelsize=14) 


ax[2].plot(Lsbin[Lsbin>0],distances_night[Lsbin>0],'.',color='midnightblue',label='Nightime')
ax[2].plot(Lsbin[Lsbin>0],distances_day[Lsbin>0],'.', color='dodgerblue',label='Daytime')
ax[2].plot(Lsbin[Lsbin>0],distances[Lsbin>0],'.',color='indianred',label='Full Sol')

ax[2].legend(fontsize=14,ncol=3)
ax[2].set_xlim(0,360)
ax[2].set_ylim(0,2)


ax[3].plot(Tau_vec["L_s"][(Tau_vec["Decimal sol"]<768) & (Tau_vec["Decimal sol"]>100)],Tau_vec["Tau"][(Tau_vec["Decimal sol"]<768) & (Tau_vec["Decimal sol"]>100)],'.',color='maroon',label='Year 1')
ax[3].plot(Tau_vec["L_s"][(Tau_vec["Decimal sol"]>768) & (Tau_vec["Decimal sol"]<1436)],Tau_vec["Tau"][(Tau_vec["Decimal sol"]>768) & (Tau_vec["Decimal sol"]<1436)],'.',color='cornflowerblue',label='Year 2')

ax[3].set_xlim(0,360)
ax[3].set_ylim(0.35,1.5)
ax[3].legend(fontsize=14,ncol=2)


ax[0].set_title("Year 1", size = 16)
ax[1].set_title("Year 2", size = 16)


ax[3].set_xlabel("Ls (°)", size = 16)

ax[0].set_ylabel("Wind Speed (m/s)", size = 16)
ax[1].set_ylabel("Wind Speed (m/s)", size = 16)

ax[2].set_ylabel("Distance metric", size = 16)
ax[3].set_ylabel("Optical depth", size = 16)


ax[0].tick_params(axis='both', labelsize=14)
ax[1].tick_params(axis='both', labelsize=14)
ax[2].tick_params(axis='both', labelsize=14)
ax[3].tick_params(axis='both', labelsize=14)

