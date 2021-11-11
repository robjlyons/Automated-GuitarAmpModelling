let pNames = [ "000", "025", "050", "075", "100" ]
let pVals = [ "0.0", "0.25", "0.50", "0.75", "1.00" ]
let NameStart = "./Recordings/DS1/20211105_LPB1_"
let trainClean = "_Training_Clean.wav"
let trainTarget = "_Training_Dirty.wav"
let testClean= "_Test_Clean.wav"
let testTarget= "_Test_Dirty.wav"
let totParams = 5



call append("$", "{")
call append("$", "\t\"Number of Parameters\":2,")
call append("$", "\t\"Data Sets\":[")
for p1 in range(0,totParams-1)
	for p2 in range(0,totParams-1)
		call append("$", "{")
		call append("$", printf("\t\"Parameters\": [ %s, %s ],", pVals[p1], pVals[p2]))
		call append("$", printf("\t\"TrainingClean\": \"%s%s_%s%s\",", NameStart, pNames[p1], pNames[p2], trainClean))
		call append("$", printf("\t\"TrainingTarget\": \"%s%s_%s%s\",", NameStart, pNames[p1], pNames[p2], trainTarget))
		call append("$", printf("\t\"TestClean\": \"%s%s_%s%s\",", NameStart, pNames[p1], pNames[p2], testClean))
		call append("$", printf("\t\"TestTarget\": \"%s%s_%s%s\"", NameStart, pNames[p1], pNames[p2], testTarget))
		call append("$", "},")
	endfor
endfor

normal G$x
call append("$", "]")
call append("$", "}")
