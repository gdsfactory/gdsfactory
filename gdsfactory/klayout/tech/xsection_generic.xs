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
h_ge = 0.4

h_etch1 = 0.07
h_etch2 = 0.06  # 60nm etch after 70nm = 130nm etch (90nm slab)
h_etch3 = 0.09  # etches the remaining 90nm slab for strip straights

h_oxide_si = 0.6
h_metal1   = 1.0
h_oxide_m1 = 2.0
h_metalh   = 0.2
h_oxide_mh = 0.5
h_metal2   = 2.0
h_ox_m2    = 0.4

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
l_npp = layer("24/0")
l_p   = layer("21/0")
l_ppp = layer("25/0")
l_PDPP  = layer("27/0")
l_bottom_implant = l_PDPP

l_Ge    = layer("30/0")
l_GeP  = layer("27/0")
l_GeN  = layer("26/0")
l_top_implant = l_GeP.or(l_GeN)


################ back-end
l_viac  = layer("40/0")
l_m1    = layer("41/0")
l_mh    = layer("47/0")
l_via1  = layer("44/0")
l_m2    = layer("45/0")
l_open  = layer("46/0")


################ stack
substrate = bulk
box = deposit(h_box)
si = deposit(h_si)

################ silicon etch to for the passives
mask(l_wg_etch1).etch(h_etch1, 0.0, :mode => :round, :into => [si]) # 70nm etch for GC, rib and strip
mask(l_wg_etch2).etch(h_etch2, 0.0, :mode => :round, :into => [si]) # 60nm etch after 70nm = 130nm etch (90nm slab)
mask(l_wg_etch3).etch(h_etch3, 0.0, :mode => :round, :into => [si]) # etches the remaining 90nm slab for strip straights

output("300/0",box)
output("301/0",si)

############### doping
mask(l_bottom_implant).etch(h_si, 0.0, :mode => :round, :into => [si])
bottom_implant = mask(l_bottom_implant).grow(h_si, 0.0, :mode => :round)

mask(l_n).etch(h_slab, 0.0, :mode => :round, :into => [si])
n = mask(l_n).grow(h_slab, 0.0, :mode => :round)

mask(l_p).etch(h_slab, 0.0, :mode => :round, :into => [si])
p = mask(l_p).grow(h_slab, 0.0, :mode => :round)

mask(l_npp).etch(h_slab, 0.0, :mode => :round, :into => [n, si, bottom_implant])
npp = mask(l_npp).grow(h_slab, 0.0, :mode => :round)

mask(l_ppp).etch(h_slab, 0.0, :mode => :round, :into => [p, si, bottom_implant])
ppp = mask(l_ppp).grow(h_slab, 0.0, :mode => :round)


output("327/0",bottom_implant)
output("330/0",p)
output("320/0",n)
output("321/0",npp)
output("331/0",ppp)


################ Ge
Ge = mask(l_Ge).grow(h_ge, 0, :bias => 0.0 , :taper => 10) #:mode => :round
output("315/0", Ge)

################# back-end
################# VIAC and M1
ox_si = deposit(h_oxide_si, h_oxide_si, :mode => :round)

mask(l_viac).etch(h_oxide_si, :taper => 4, :into => [ox_si])
viac = deposit(h_metal1, h_metal1)
mask(l_m1.inverted()).etch(h_metal1 + h_metal1, :taper => 4, :into => [viac])

ox_m1 = deposit(h_oxide_m1, h_oxide_m1, :mode => :round)
planarize(:less=>0.9, :into=>[ox_m1])


output("302/0", ox_si)
output("303/0", viac)
output("307/0", ox_m1)

################# MH
mh = deposit(h_metalh, h_metalh)
mask(l_mh.inverted()).etch(h_metalh + h_metalh, :taper => 4, :into => [mh])
output("306/0",mh)

ox_mh = deposit(h_oxide_mh, h_oxide_mh, :mode => :round)
output("317/0",ox_mh)

################# VIA1 and M2
mask(l_via1).etch(h_oxide_mh + h_oxide_m1, :taper => 4, :into => [ox_mh,ox_m1])
via1 = deposit(h_metal2, 1.5*h_metal2)
mask(l_m2.inverted()).etch(h_metal2, :taper => 4, :into => [via1])
output("308/0",via1)

################# passivation and ML Open
ox_m2 = deposit(h_ox_m2, h_ox_m2, :mode => :round)
mask(l_open).etch(h_ox_m2 + h_ox_m2, :into=>[ox_m2], :taper=>5) # etch oxide open
output("309/0",ox_m2)
