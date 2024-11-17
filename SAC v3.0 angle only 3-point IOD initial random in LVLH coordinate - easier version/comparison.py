import orekit
vm = orekit.initVM()
#orekit需要初始化才能使用
from orekit.pyhelpers import setup_orekit_curdir
#每次用orekit要带着orekit-data.zip
setup_orekit_curdir()
# 导入所需的类和模块
from org.orekit.orbits import KeplerianOrbit,OrbitType, PositionAngle,CartesianOrbit
#KeplerianOrbit根据开普勒轨道根数生成轨道
#OrbitType指定不同类型轨道参数
#PositionAngle表示航天器在轨道上的位置，即偏近点角(eccentric)、平近点角(mean)、真近点角(true)三种表达方式
from org.orekit.frames import FramesFactory #坐标系
from org.orekit.time import TimeScalesFactory, AbsoluteDate
#TimeScaleFactory引入时间基准
#AbsoluteDate表达某一时间基准下的某一时刻
from org.orekit.propagation import SpacecraftState
#spacecraftstate可以以多种方式生成航天器状态
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
#积分器，需要手动设定最小步长（秒）、最大步长（秒）、位置误差（米）
from org.orekit.propagation.numerical import NumericalPropagator
#numericalpropagator使用数值积分方法外推轨道，因此需要指定积分器，精度比分析预报器高，需要指定若干参数
# “Numerical propagation is much more accurate than analytical propagation”
from org.orekit.utils import IERSConventions,PVCoordinatesProvider,Constants,PVCoordinates
#Constants：常数库
#IERS规范坐标系
#PV坐标提供器：position和velocity
from org.orekit.bodies import OneAxisEllipsoid,CelestialBodyFactory
#CelestialBodyFactory:天体数据库
#OneAxisEllipsoid天体定义为单轴椭球
from org.orekit.forces.gravity.potential import GravityFieldFactory #重力场模型集合
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel,ThirdBodyAttraction
#一种高阶次重力场计算方法
#三体引力
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
#solarradiationpressure是太阳光压
#IsotropicRadiationSingleCoefficient是简化的受光压航天器模型
from org.hipparchus.geometry.euclidean.threed import Vector3D
#用于生成三维冲量
from org.orekit.attitudes import InertialProvider
#无参数时固定给出J2000坐标系下姿态
from org.orekit.propagation.events import DateDetector
#到达给定时间时触发
from org.orekit.forces.maneuvers import ImpulseManeuver
#脉冲变轨

from math import radians #角度转弧度;反余弦;弧度转角度
import numpy as np

def J2000_Kepler_prop_to_Cartesian(a,e,i,raan,argument_of_periapsis,true_anomoly,time_interval,year=2023,month=9,day=1,hour=0,minute=0,sec=0.000,light_pressrue=True,Sun=True,Moon=True,grav_deg=6):
    mu = Constants.WGS84_EARTH_MU    #WGS84地球引力常数
    utc = TimeScalesFactory.getUTC() #UTC时间
    J2000=FramesFactory.getEME2000() #EME2000就是J2000；约定俗成的系都在FramesFactory里
    initial_date=AbsoluteDate(year,month,day,hour,minute,sec,utc) #指定绝对时间时需要指定基准
    initialorbit=KeplerianOrbit(float(a*1000),float(e),float(radians(i)),float(radians(argument_of_periapsis)),float(radians(raan)),float(radians(true_anomoly)),
                             PositionAngle.TRUE,
                             J2000,initial_date,mu) #PositionAngle.True指定用真近点角表示位置
    #定义目标初始轨道，单位都是国际单位制基本单位；角度国际单位制是弧度，所以用radians转成弧度
    #定义操控星初始轨道
    mass=1000.0
    initstate=SpacecraftState(initialorbit,mass) #轨道+质量定义卫星

    #定义积分器
    minStep = 0.001
    maxstep = 1000.0
    absTol  = 1e-6
    relTol  = 1e-6
    initStep = 60.0
    integrator = DormandPrince853Integrator(minStep, maxstep, absTol, relTol) #absTol和relTol是容忍度
    integrator.setInitialStepSize(initStep) #设定初始步长
    propagator_num=NumericalPropagator(integrator) #numericalpropagator函数里只需要指定积分器+姿态角提供器，其他参数用setXXX提供
    propagator_num.setOrbitType(OrbitType.CARTESIAN) #输出数据的类型

    #为了发挥numerical interegator的高精度优势，定义各种摄动力，如椭球、光压等；地球的各种参数定义在ITRF里，所以没用J2000
    ITRF=FramesFactory.getITRF(IERSConventions.IERS_2010,True) #True是处理EOP（地球指向参数）混合情况用的
    earth=OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,Constants.WGS84_EARTH_FLATTENING,ITRF)
    #在ITRF坐标系下，用地球赤道半径与扁率定义椭球
    gravityProvider=GravityFieldFactory.getNormalizedProvider(grav_deg,grav_deg)
    #从文件中读取重力场，Get the gravity field coefficients provider from the first supported file.
    propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(ITRF,gravityProvider))
    #用H-F模型定义重力场，需要参考系（ITRF）和重力场模型（自动考虑了J2摄动）
    solarRadiationPressureCoefficient = 1.0
    #太阳光压系数
    solarRadiationPressureArea = 0.025
    #光压面积
    radiationSensitive = IsotropicRadiationSingleCoefficient(solarRadiationPressureArea,solarRadiationPressureCoefficient)
    #用光压面积与光压系数定义简单的航天器光压模型
    SunPVProvider = PVCoordinatesProvider.cast_(CelestialBodyFactory.getSun())
    #从天体库得到太阳位置
    # .cast_:java是强类型语言，python为动态类型，因此偶尔需要指定引用对象的类型/接口
    srp = SolarRadiationPressure(SunPVProvider,earth.getEquatorialRadius(),radiationSensitive)
    #定义光压需要太阳位置、赤道半径和航天器感光情况；需要地球半径以计算地影区、半影区
    if(light_pressrue):
        propagator_num.addForceModel(srp)
    #在预报器中增加太阳光压项

    # 三体引力
    thirdBody_sun   = ThirdBodyAttraction(CelestialBodyFactory.getSun())
    thirdBody_moon  = ThirdBodyAttraction(CelestialBodyFactory.getMoon())
    if(Sun):
        propagator_num.addForceModel(thirdBody_sun)
    if(Moon):
        propagator_num.addForceModel(thirdBody_moon)

    #为了确保目标星与追击星状态数组大小一致，把脉冲提到前面来
    date = initial_date.shiftedBy(float(time_interval))
    # 预测机动点PV
    propagator_num.setInitialState(initstate)
    new_state = propagator_num.propagate(date)
    new_pos = new_state.getPVCoordinates().getPosition()
    new_vel = new_state.getPVCoordinates().getVelocity()
    new_pos = np.array([new_pos.getX(), new_pos.getY(), new_pos.getZ()])
    new_vel = np.array([new_vel.getX(), new_vel.getY(), new_vel.getZ()])
    return new_pos,new_vel