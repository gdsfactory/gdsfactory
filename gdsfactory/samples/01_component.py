"""Based on phidl tutorial.

We'll start by assuming we have a function straight() which already exists
and makes us a simple straight rectangle. Many functions like this
exist in the gdsfactory.components library and are ready-for-use.
We write this one out fully just so it's explicitly clear what's happening

"""

import gdsfactory as gf


@gf.cell
def straight_sample(length=5, width=1):
    wg = gf.Component("straight_sample")
    wg.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=(2, 0))
    wg.add_port(name="o1", midpoint=[0, width / 2], width=width, orientation=180)
    wg.add_port(name="o2", midpoint=[length, width / 2], width=width, orientation=0)
    return wg


# ==============================================================================
# Create a blank component
# ==============================================================================
# Let's create a new Component ``c`` which will act as a blank canvas (c can be
# thought of as a blank GDS cell with some special features). Note that when we
# make a Component

if __name__ == "__main__":
    c = gf.Component("MultiWaveguide")

    # Now say we want to add a few straights to to our  Component" c.
    # First we create the straights.  As you can see from the straight() function
    # definition, the straight() function creates another Component ("WG").
    # This can be thought of as the straight() function creating another GDS cell,
    # only this one has some geometry inside it.
    #
    # Let's create two of these Devices by calling the straight() function
    WG1 = straight_sample(length=10, width=1)
    WG2 = straight_sample(length=12, width=2)

    # Now we've made two straights Component WG1 and WG2, and we have a blank
    # Component c. We can add references from the devices WG1 and WG2 to our blank
    # Component by using the add_ref() function.
    # After adding WG1, we see that the add_ref() function returns a handle to our
    # reference, which we will label with lowercase letters wg1 and wg2.  This
    # handle will be useful later when we want to move wg1 and wg2 around in c.
    wg1 = c.add_ref(WG1)  # Using the function add_ref()
    wg2 = c << WG2  # Using the << operator which is identical to add_ref()

    # Alternatively, we can do this all on one line
    wg3 = c.add_ref(straight_sample(length=14, width=3))

    c.show()  # show it in Klayout
