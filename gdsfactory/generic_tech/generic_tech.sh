export PDK_ROOT=$PWD
export KLAYOUT_HOME="$PDK_ROOT/klayout"

klayout -e -j $KLAYOUT_HOME -l $KLAYOUT_HOME/tech/generic_tech.lyp -nn $KLAYOUT_HOME/tech/generic_tech.lyt
