#######################
# cross-section settings script for Klayout using generic layer map
# https://sourceforge.net/p/xsectionklayout/wiki/MainPage/
# Download the ruby module here: http://sourceforge.net/p/xsectionklayout/code/HEAD/tree/trunk/src/xsection.rbm?format=raw
# Copy xsection.lym to the installation path of KLayout macros. That is the place where the KLayout binary is installed
# ~/.klayout/macros/ (in Mac)
# C:\Users\username\AppData\Roaming\KLayout (64bit) (in windows)
#######################

# Sample technology values, modify them to match your particular technology

h_box = 1.0
h_slab = 0.09
h_si = 0.22
hge = 0.4
hnitride = 0.4

h_etch1 = 0.07
h_etch2 = 0.06  # 60nm etch after 70nm = 130nm etch (90nm slab)
h_etch3 = 0.09  # etches the remaining 90nm slab for strip waveguides

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

delta(dbu) # Declare the basic accuracy used to remove artifacts for example: delta(5 * dbu)
depth(12.0)
height(12.0)

################ front-end
l_wg    = layer("1/0")
l_fc    = layer("2/0")
l_rib   = layer("3/0")

l_wg_etch1  = l_wg.inverted()            # protects ridge
l_wg_etch2  = (l_fc.or(l_wg)).inverted() # protects ridge and grating couplers from the etch down to the slab (forms rib waveguides)
l_wg_etch3  = (l_rib.or(l_fc).or(l_wg)).inverted()  # protects ridge, grating couplers and rib waveguides from the final etch to form strip waveguides

l_n   = layer("20/0")
l_np  = layer("22/0")
l_npp = layer("24/0")
l_p   = layer("21/0")
l_pp  = layer("23/0")
l_ppp = layer("25/0")
l_PDPp  = layer("27/0")
l_bottom_implant = l_PDPp

l_nitride  = layer("34/0")
l_Ge    = layer("30/0")
l_GePpp  = layer("42/0")
l_GeNpp  = layer("24/0")
l_top_implant = l_GePpp.or(l_GeNpp)

################ back-end
l_via1  = layer("40/0")
l_m1    = layer("41/0")
l_mh    = layer("47/0")
l_via2  = layer("44/0")
l_m2    = layer("45/0")
l_via3  = layer("43/0")
l_m3    = layer("49/0")
l_open  = layer("46/0")

################ stack
substrate = bulk
box = deposit(h_box)
si = deposit(h_si)

################ silicon etch to for the passives
mask(l_wg_etch1).etch(h_etch1, 0.0, :mode => :round, :into => [si]) # 70nm etch for GC, rib and strip
mask(l_wg_etch2).etch(h_etch2, 0.0, :mode => :round, :into => [si]) # 60nm etch after 70nm = 130nm etch (90nm slab)
mask(l_wg_etch3).etch(h_etch3, 0.0, :mode => :round, :into => [si]) # etches the remaining 90nm slab for strip waveguides

output("300/0",box)
output("301/0",si)

############### doping
mask(l_bottom_implant).etch(h_si, 0.0, :mode => :round, :into => [si])
bottom_implant = mask(l_bottom_implant).grow(h_si, 0.0, :mode => :round)

mask(l_n).etch(h_slab, 0.0, :mode => :round, :into => [si])
n = mask(l_n).grow(h_slab, 0.0, :mode => :round)

mask(l_p).etch(h_slab, 0.0, :mode => :round, :into => [si])
p = mask(l_p).grow(h_slab, 0.0, :mode => :round)

mask(l_np).etch(h_slab, 0.0, :mode => :round, :into => [n, p, si, bottom_implant])
np = mask(l_np).grow(h_slab, 0.0, :mode => :round)

mask(l_pp).etch(h_slab, 0.0, :mode => :round, :into => [n, p, si, bottom_implant])
pp = mask(l_pp).grow(h_slab, 0.0, :mode => :round)

mask(l_npp).etch(h_slab, 0.0, :mode => :round, :into => [n, p, np, pp, si, bottom_implant])
npp = mask(l_npp).grow(h_slab, 0.0, :mode => :round)

mask(l_ppp).etch(h_slab, 0.0, :mode => :round, :into => [n, p, np, pp, si, bottom_implant])
ppp = mask(l_ppp).grow(h_slab, 0.0, :mode => :round)

output("327/0",bottom_implant)
output("330/0",p)
output("320/0",n)
output("321/0",npp)
output("331/0",ppp)

################ Ge
Ge = mask(l_Ge).grow(hge, 0, :bias => 0.0 , :taper => 10)
output("315/0", Ge)

################ Nitride
ox_nitride = deposit(2*hoxidenitride, 2*hoxidenitride)
planarize(:less=> hoxidenitride, :into=>[ox_nitride])
output("302/0", ox_nitride)
nitride = mask(l_nitride).grow(hnitride, 0, :bias => 0.0 , :taper => 10)
output("305/0", nitride)

################# back-end
################# VIA1, M1 and MH
ox_si = deposit(hoxidesi + hge + hnitride, hoxidesi + hge + hnitride, :mode => :round)
planarize(:less=> hge + hnitride, :into=>[ox_si])
mask(l_via1).etch(hoxidesi + hge + hnitride + hoxidenitride, :taper => 4, :into => [ox_si, ox_nitride])

via1 = deposit(2*hoxidesi, 2* hoxidesi)
planarize(:less=> 2*hoxidesi, :into=>[via1])

mh = deposit(hheater, hheater)
mask(l_mh.inverted()).etch(hheater + hheater, :into => [mh])
m1 = deposit(hm1, hm1)
mask(l_m1.inverted()).etch(hm1 + hm1, :into => [m1])
output("306/0", mh)
output("399/0", m1)

output("302/0", ox_si)
output("303/0", via1)

################# VIA2 and M2
ox_m1 = deposit(2*hoxidem1, 2*hoxidem1, :mode => :round)
planarize(:less=>hoxidem1, :into=>[ox_m1])

mask(l_via2).etch(hoxidem1 + hm1m2, :taper => 4, :into => [ox_m1])
via2 = deposit(hm2, hm2)

mask(l_m2.inverted()).etch(hm2, :taper => 4, :into => [via2])
output("308/0",via2)

ox_m2 = deposit(2*hoxidem2, 2*hoxidem2, :mode =>:round)
planarize(:less=>hoxidem2, :into=>[ox_m2])
output("309/0", ox_m2)
output("307/0", ox_m1)

################# VIA3 and M3
mask(l_via3).etch(hoxidem2 + hm2m3 , :taper => 4, :into => [ox_m2, ox_m2])
via3 = deposit(hm3, hm3)
mask(l_m3.inverted()).etch(hm3, :taper => 4, :into => [via3])
output("310/0",via3)

################# passivation and ML Open
ox_m3 = deposit(hoxidem3, hoxidem3, :mode => :round)
mask(l_open).etch(hoxidem3 + hoxidem3, :into=>[ox_m3], :taper=>5) # etch oxide open
output("311/0",ox_m3)
