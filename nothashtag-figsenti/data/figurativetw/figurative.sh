##### grep huge twitter 2013 dump
zgrep -i "#sarcas" 2013/*/tweet*.en.id_text.gz > sarcas.txt
zgrep -E -i "#iron(y|ic)" 2013/*/tweet*.en.id_text.gz > ironic.txt
zgrep -i "#not" 2013/*/tweet*.en.id_text.gz > not.txt
zgrep -iw "literally" 2013/*/tweet*.en.id_text.gz > literally.txt
zgrep -iw "virtually" 2013/*/tweet*.en.id_text.gz > virtually.txt
zgrep -i "#yeahright" 2013/*/tweet*.en.id_text.gz > yeahright.txt
zgrep -i "Oh.*you must" 2013/*/tweet*.en.id_text.gz > ohyoumust.txt
zgrep -i "\bas .* as\b" 2013/*/tweet*.en.id_text.gz > asXas.txt
zgrep -i "so to speak" 2013/*/tweet*.en.id_text.gz > sotospeak.txt
zgrep -i "don't you love" 2013/*/tweet*.en.id_text.gz > dontyoulove.txt
zgrep -iw "proverbial" 2013/*/tweet*.en.id_text.gz > proverbial.txt
zgrep -i "#justkidding" 2013/*/twee*en.id_text.gz > justkidding.txt