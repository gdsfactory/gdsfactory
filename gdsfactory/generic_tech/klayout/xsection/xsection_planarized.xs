#######################
# cross-section settings script for Klayout using generic layer map
# https://sourceforge.net/p/xsectionklayout/wiki/MainPage/
# Download the ruby module here: http://sourceforge.net/p/xsectionklayout/code/HEAD/tree/trunk/src/xsection.rbm?format=raw
# Copy xsection.lym to the installation path of KLayout macros. That is the place where the KLayout binary is installed
# ~/.klayout/macros/ (in Mac)
# C:\Users\username\AppData\Roaming\KLayout (64bit) (in windows)
#######################

# Sample technology values, modify them to match your particular technology

t_box = 1.0
t_slab = 0.09
t_si = 0.22
t_ge = 0.4
t_nitride = 0.4

h_etch1 = 0.07
h_etch2 = 0.06  # 60nm etch after 70nm = 130nm etch (90nm slab)
h_etch3 = 0.09  # etches the remaining 90nm slab for strip straights

hsim1 = 0.8
hm1   = 0.5
hm1m2 = 0.6
hm2m3    = 0.3
hheater   = 0.1
hoxidesi = 0.6
hoxidenitride = 0.1
hoxidem1 = 0.6
hoxidem2 = 2.0
hoxidem3 = 0.5
hm2   = 0.5
hm3   = 2.0

# Declare the basic accuracy used to remove artifacts for example: delta(5 * dbu)
delta(dbu)
depth(12.0)
height(12.0)

################ front-end
l_wg    = layer("1/0")
l_fc    = layer("2/0")
l_rib   = layer("3/0")

l_wg_etch1  = l_wg.inverted()            # protects ridge
l_wg_etch2  = (l_fc.or(l_wg)).inverted() # protects ridge and grating couplers from the etch down to the slab (forms rib straights)
l_wg_etch3  = (l_rib.or(l_fc).or(l_wg)).inverted()  # protects ridge, grating couplers and rib straights from the final etch to form strip straights

l_n   = layer("20/0")
l_np  = layer("22/0")
l_npp = layer("24/0")
l_p   = layer("21/0")
l_pp  = layer("23/0")
l_ppp = layer("25/0")
l_PDPP  = layer("27/0")
l_bottom_implant = l_PDPP

l_nitride  = layer("34/0")
l_Ge    = layer("30/0")
l_GePPp  = layer("42/0")
l_GeNPP  = layer("24/0")
l_top_implant = l_GePPp.or(l_GeNPP)

################ back-end
l_viac  = layer("40/0")
l_m1    = layer("41/0")
l_mh    = layer("47/0")
l_via1  = layer("44/0")
l_m2    = layer("45/0")
l_via2  = layer("43/0")
l_m3    = layer("49/0")
l_open  = layer("46/0")

################ stack
substrate = bulk
box = deposit(t_box)
si = deposit(t_si)

################ silicon etch to for the passives
mask(l_wg_etch1).etch(h_etch1, 0.0, :mode => :round, :into => [si]) # 70nm etch for GC, rib and strip
mask(l_wg_etch2).etch(h_etch2, 0.0, :mode => :round, :into => [si]) # 60nm etch after 70nm = 130nm etch (90nm slab)
mask(l_wg_etch3).etch(h_etch3, 0.0, :mode => :round, :into => [si]) # etches the remaining 90nm slab for strip straights

output("300/0",box)
output("301/0",si)

############### doping
mask(l_bottom_implant).etch(t_si, 0.0, :mode => :round, :into => [si])
bottom_implant = mask(l_bottom_implant).grow(t_si, 0.0, :mode => :round)

mask(l_n).etch(t_slab, 0.0, :mode => :round, :into => [si])
n = mask(l_n).grow(t_slab, 0.0, :mode => :round)

mask(l_p).etch(t_slab, 0.0, :mode => :round, :into => [si])
p = mask(l_p).grow(t_slab, 0.0, :mode => :round)

mask(l_np).etch(t_slab, 0.0, :mode => :round, :into => [n, p, si, bottom_implant])
np = mask(l_np).grow(t_slab, 0.0, :mode => :round)

mask(l_pp).etch(t_slab, 0.0, :mode => :round, :into => [n, p, si, bottom_implant])
pp = mask(l_pp).grow(t_slab, 0.0, :mode => :round)

mask(l_npp).etch(t_slab, 0.0, :mode => :round, :into => [n, p, np, pp, si, bottom_implant])
npp = mask(l_npp).grow(t_slab, 0.0, :mode => :round)

mask(l_ppp).etch(t_slab, 0.0, :mode => :round, :into => [n, p, np, pp, si, bottom_implant])
ppp = mask(l_ppp).grow(t_slab, 0.0, :mode => :round)

output("327/0",bottom_implant)
output("330/0",p)
output("320/0",n)
output("321/0",npp)
output("331/0",ppp)

################ Ge
Ge = mask(l_Ge).grow(t_ge, 0, :bias => 0.0 , :taper => 10)
output("315/0", Ge)

################ Nitride
ox_nitride = deposit(2*hoxidenitride, 2*hoxidenitride)
planarize(:less=> hoxidenitride, :into=>[ox_nitride])
output("302/0", ox_nitride)
nitride = mask(l_nitride).grow(t_nitride, 0, :bias => 0.0 , :taper => 10)
output("305/0", nitride)

################# back-end
################# VIAC, M1 and MH
ox_si = deposit(hoxidesi + t_ge + t_nitride, hoxidesi + t_ge + t_nitride, :mode => :round)
planarize(:less=> t_ge + t_nitride, :into=>[ox_si])
mask(l_viac).etch(hoxidesi + t_ge + t_nitride + hoxidenitride, :taper => 4, :into => [ox_si, ox_nitride])

viac = deposit(2*hoxidesi, 2* hoxidesi)
planarize(:less=> 2*hoxidesi, :into=>[viac])

mh = deposit(hheater, hheater)
mask(l_mh.inverted()).etch(hheater + hheater, :into => [mh])
m1 = deposit(hm1, hm1)
mask(l_m1.inverted()).etch(hm1 + hm1, :into => [m1])
output("306/0", mh)
output("399/0", m1)

output("302/0", ox_si)
output("303/0", viac)

################# VIA1 and M2
ox_m1 = deposit(2*hoxidem1, 2*hoxidem1, :mode => :round)
planarize(:less=>hoxidem1, :into=>[ox_m1])

mask(l_via1).etch(hoxidem1 + hm1m2, :taper => 4, :into => [ox_m1])
via1 = deposit(hm2, hm2)

mask(l_m2.inverted()).etch(hm2, :taper => 4, :into => [via1])
output("308/0",via1)

ox_m2 = deposit(2*hoxidem2, 2*hoxidem2, :mode =>:round)
planarize(:less=>hoxidem2, :into=>[ox_m2])
output("309/0", ox_m2)
output("307/0", ox_m1)

################# VIA2 and M3
mask(l_via2).etch(hoxidem2 + hm2m3 , :taper => 4, :into => [ox_m2, ox_m2])
via2 = deposit(hm3, hm3)
mask(l_m3.inverted()).etch(hm3, :taper => 4, :into => [via2])
output("310/0",via2)

################# passivation and ML Open
ox_m3 = deposit(hoxidem3, hoxidem3, :mode => :round)
mask(l_open).etch(hoxidem3 + hoxidem3, :into=>[ox_m3], :taper=>5)
output("311/0",ox_m3)
