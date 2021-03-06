var totalCities = -1;

var cities = new Array();
var order = new Array();

var bestOrderA = new Array();
var bestCostA = 0;
var bestOrderB = new Array();
var bestCostB = 0;

var greedy_list = new Array();

var pre_cost = new Array();
var next_cost = new Array();

var esets = new Array();
var inter_edges = new Array();

var Eset = new Array();
var intermediate = new Array();
var next_generation = new Array();

var show_e = true;
var show_i = true;
var show_n = true;

function set(){
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.beginPath();

	while(bestOrderA.length > 0)
		bestOrderA.pop();
	bestCostA = 0.0
	while(bestOrderB.length > 0)
		bestOrderB.pop();
	bestCostB = 0.0

	while(pre_cost.length > 0)
		pre_cost.pop();
	while(next_cost.length > 0)
		next_cost.pop();

	while(esets.length > 0)
		esets.pop();
	while(inter_edges.length > 0)
		inter_edges.pop();

	while(Eset.length > 0)
		Eset.pop();
	while(intermediate.length > 0)
		intermediate.pop();
	while(next_generation.length > 0)
		next_generation.pop();
		
	document.getElementById("costPa").innerHTML = "Cost (ParentA): ";
	document.getElementById("costPb").innerHTML = "Cost (ParentB): ";
	document.getElementById("costO1").innerHTML = "Cost (Offspring1): ";
	document.getElementById("costO2").innerHTML = "Cost (Offspring2): ";
	document.getElementById("costO3").innerHTML = "Cost (Offspring3): ";
	
	show_e = true;
	show_i = true;
	show_n = true;
}

function reset(flg){
	if(flg == 0){
		totalCities = document.getElementById("inputCt").value;
		
		while(cities.length > 0)
			cities.pop();
		while(order.length > 0)
			order.pop();
	}
	
	set();
	
	if(totalCities < 6){
		if(totalCities < 3)
			return;
		
		if(flg == 0){
			createCities(totalCities);
			while(greedy_list.length > 0)
				greedy_list.pop();
			for(var i = 0; i < totalCities; i++)
				greedy_list.push(new Array());
		}
		
		keepCities();
		greedy(order);
		var min = Infinity;
		var x = -1;
		var val = -1;
		for(var i = 0; i < greedy_list.length; i++){
			val = calcost(greedy_list[i]);
			if(min > val){
				min = val;
				x = i;
			}
		}
		
		drawPath(greedy_list[x]);
		document.getElementById("costPa").innerHTML += min;
		
		return;
	}
	
	if(flg == 0)
		createCities(totalCities);
		while(greedy_list.length > 0)
			greedy_list.pop();
		for(var i = 0; i < totalCities; i++)
			greedy_list.push(new Array());
	
	greedy(order);
	mergePath(bestOrderA,bestOrderB);
	while(Eset.length == 0)
		run();
	
	for(var i = 0; i < Eset.length; i++)
		intermediate.push(inter_sol(bestOrderA,Eset[i]));
		
	var temp = new Array();
	for(var i = 0; i < Eset.length; i++){
		temp = new_connection(inter_sol(bestOrderA,Eset[i]));
		next_generation.push(temp.slice());
		esets.push([Eset[i].slice()]);
		
		for(var j = 0; j < next_generation.length; j++){
			if(add(next_generation[j],Eset[i])){
				next_generation[j] = new_connection(inter_sol(next_generation[j],Eset[i]));
				esets[j].push(Eset[i].slice());
			}
		}
	}
	
	for(var i = 0; i < next_generation.length; i++)
		next_cost.push(calcost(next_generation[i]));
		
	if(next_cost.length > 3){
		var min = Infinity;
		var max = -Infinity;
		var a = -1;
		var b = -1;
		
		for(var i = 0; i < next_cost.length; i++){
			if(min > next_cost[i]){
				min = next_cost[i];
				a = i;
			}
			if(i < 3 && max < next_cost[i]){
				max = next_cost[i];
				b = i;
			}
		}
		
		if(min > bestCostA || min > bestCostB)	reset(1);
		else{
			var c = intermediate[a].slice();
			intermediate[a] = intermediate[b].slice();
			intermediate[b] = c;
			
			var d = next_generation[a].slice();
			next_generation[a] = next_generation[b].slice();
			next_generation[b] = d;
			
			var e = next_cost[a];
			next_cost[a] = next_cost[b];
			next_cost[b] = e;
			
			var f = esets[a].slice();
			esets[a] = esets[b].slice();
			esets[b] = f;
		}					
	}
}

function run(){
	set();
	
	keepCities();
	greedy(order);
	mergePath(bestOrderA,bestOrderB);
	Eset = mergeCycles(bestOrderA,bestOrderB).slice();
}

function add(arr,e){
	for(var i = 0; i < e.length-1; i += 2){
		var a = e[i];
		var b = e[i+1];
		var c = -1;
		for(var j = 0; j < arr.length-1; j++)
			if((a == arr[j] && b == arr[j+1]) || (a == arr[j+1] && b == arr[j]))
				c = 1;
		if(c == -1)
			return false;
	}
	return true;
}

function calcost(arr){
	var sum = 0;
	for(var i = 0; i < order.length-1; i++){
		var x = cities[arr[i]][0] - cities[arr[i+1]][0];
		var y = cities[arr[i]][1] - cities[arr[i+1]][1];
		sum += Math.pow((x*x + y*y),0.5);
	}
	return sum;
}

function calDistance(a,b){
	var x = cities[a][0] - cities[b][0];
	var y = cities[a][1] - cities[b][1];
	return Math.pow((x*x + y*y),0.5);
}

function greedy(arr){
	var n = arr.length;
	var candi = new Array();
	var tmp = new Array();
	
	var a = Math.floor(Math.random() * (n-1));
	var b = Math.floor(Math.random() * (n-1));
	
	var randCh = [a,b];
	
	for(var i = 0; i < 2; i++){
		if(greedy_list[randCh[i]].length != 0)
			continue;
			
		var chk = new Array();
		for(var j = 0; j < n-1; j++)
			chk.push(0);
		chk[randCh[i]] = 1;
		candi.push(arr[randCh[i]]);

		while(true){
			var min = Infinity;
			var nextv = -1;
			for(var j = 0; j < n-1; j++){
				if(candi[candi.length-1] == arr[j])
					continue;
				
				if(chk[j] == 0){
					var temp = calDistance(candi[candi.length-1],arr[j]);
					if(min > temp){
						min = temp;
						nextv = j;
					}
				}
			}
			
			candi.push(arr[nextv]);
			chk[nextv] = 1;
			
			var k = 0;
			while(k < n-1){
				if(chk[k] == 0)
					break;
				k++;
			}
			if(k == n-1)
				break;
		}
		k = 0;
		for(var j = 0; j < candi.length; j++)
			if(candi[j] == 0)
				k = j;
		
		for(var j = 0; j < candi.length; j++)
			tmp.push(candi[(j+k)%(candi.length)]);
			
		tmp.push(0);
		
		greedy_list[randCh[i]] = tmp.slice();
		
		while(tmp.length > 0)
			tmp.pop();
		while(candi.length > 0)
			candi.pop();
	}
	
	bestOrderA = greedy_list[a].slice();
	bestCostA = calcost(bestOrderA);
	bestOrderB = greedy_list[b].slice();
	bestCostB = calcost(bestOrderB);
	
	drawTwoPath(bestOrderA,bestOrderB);
	document.getElementById("costPa").innerHTML += bestCostA;
	document.getElementById("costPb").innerHTML += bestCostB;
}

function mergeCycles(A,B){
	var n = A.length;
	var AnB = A.concat(B);
	var x = -1;
	var ABcyc = new Array();
	var candi = new Array();
	for(var i = 0; i < n-1; i++){
		while(ABcyc.length > 0)
			ABcyc.pop();
		var prsnt = A[i];
		var nextv = A[i+1];
		var indi = 1
		
		while(nextv != A[i]){
			ABcyc.push(prsnt);
			var temp = -1;
			
			if(indi == 1)
				x = n;
			else if(indi == -1)
				x = 0;
				
			for(var j = 0; j < n-1; j++)
				if(AnB[j+x] == nextv){
					temp = j;
					break;
				}
				
			if(Math.random() < 0.5){
				if(prsnt == AnB[(temp+n-2)%(n-1)+x]){
					prsnt = nextv;
					nextv = AnB[(temp+1)%(n-1)+x];
				} else {
					prsnt = nextv;
					nextv = AnB[(temp+n-2)%(n-1)+x];
				}
			} else {
				if(prsnt == AnB[(temp+1)%(n-1)+x]){
					prsnt = nextv;
					nextv = AnB[(temp+n-2)%(n-1)+x];
				} else {
					prsnt = nextv;
					nextv = AnB[(temp+1)%(n-1)+x];
				}
			}
			indi *= -1;
		}
		ABcyc.push(prsnt);
		ABcyc.push(nextv);
		
		if(ABcyc.length % 2 == 1 && ABcyc.length < totalCities && ABcyc.length > 3 && cycle(ABcyc))
			if(!distinct(candi,ABcyc))
				candi.push(ABcyc.slice());
	}
	return candi;
}

function cycle(ABcyc){
	for(var i = 1; i < ABcyc.length-1; i++)
		for(var j = 0; j < i; j++)
			if(ABcyc[i] == ABcyc[j])
				return false;
	return true;
}

function distinct(candi,ABcyc){
	var n = ABcyc.length;
	for(var i = 0; i < candi.length; i++)
		if(n == candi[i].length){
			var temp = -1;
			for(var j = 0; j < n-1; j++)
				if(ABcyc[0] == candi[i][j]){
					temp = j;
					break;
				}
			for(var j = 0; j < n-1; j++)
				if(ABcyc[j] != candi[i][(temp+j)%(n-1)]){
					temp = -1;
					break;
				}
			if(temp  -1)
				return true;
		}
	return false;
}

function sameEdge(used,arr){
	var n = arr.length;
	if(used.length == 0)
		return false;
		
	for(var i = 0; i < n-1; i++)
		for(var j = 0; j < used.length; j++)
			for(var k = 0; k < used[j].length-1; k++){
				if(arr[i] == used[j][k] && arr[i+1] == used[j][k+1])
					return true;
				if(arr[i+1] == used[j][k] && arr[i] == used[j][k+1])
					return true;
			}
	return false;
}

function inter_conn(sub,li,chk,flg){
	var j = 0;
	while(j < li.length){
		var temp = -1;
		var a = -1;
		var b = -1;
		if(flg == 1 || flg == 2){
			temp = sub.length-1;
			a = 0;
			b = 1;
		} else if(flg == -1 || flg == -2){
			temp = 0;
			a = 1;
			b = 0;
		}
		
		if(sub[temp] == li[j][a]){
			if(chk[j] == 0 || chk[j] == -1){
				if(flg == 1 || flg == 2)
					sub.push(li[j][b]);
				else if(flg == -1 || flg == -2)
					sub.splice(0, 0, li[j][b]);
					
				chk[j] = 1;
				j = 0;
			} else
				j += 1;
		} else if(sub[temp] == li[j][b]){
			if(flg == 2 || flg == -2){
				if(chk[j] == 0 || chk[j] == -1){
					if(flg == 2)
						sub.push(li[j][a]);
					else if(flg == -2)
						sub.splice(0, 0, li[j][a]);
						
					chk[j] = 1;
					j = 0;
				} else
					j += 1;
			} else {
				if(chk[j] == -1){
					if(flg == 1)
						sub.push(li[j][a]);
					else if(flg == -1)
						sub.splice(0, 0, li[j][a]);
						
					chk[j] = 1;
					j = 0;
				} else
					j += 1;
			}
		} else
			j += 1;
	}
	return [sub, li, chk];
}

function inter_sol(A, E){
	var inter = new Array();
	var sub = new Array();
	var edges = new Array();
	var Y = A.slice();
	var n = Y.length;
	
	for(var i = 0; i < n-1; i++){
		var tmp = [A[i],A[i+1]];
		edges.push(tmp.slice());
		while(tmp.length > 0)
			tmp.pop();
	}
	
	inter_edges = edges.slice();
	
	var li = edges.slice();
	for(var j = 0; j < E.length-1; j += 2)
		for(var k = 0; k < li.length; k++){
			if(E[j] == li[k][0])
				if(E[j+1] == li[k][1]){
					var idx = li.indexOf(li[k]);
					if(idx > -1) li.splice(idx,1);
					break;
				}
			if(E[j+1] == li[k][0])
				if(E[j] == li[k][1]){
					var idx = li.indexOf(li[k]);
					if(idx > -1) li.splice(idx,1);
					break;
				}
		}
		
	var chk = new Array();
	for(var j = 0; j < li.length; j++)
		chk[j] = 0;
	
	for(var j = 1; j < E.length; j += 2){
		var edgeB = [E[j],E[j+1]];
		li.push(edgeB.slice());
		chk.push(-1);
		while(edgeB.length > 0)
			edgeB.pop();
	}
	
	sub.push(li[0][0]);
	sub.push(li[0][1]);
	
	chk[0] = 1;
	
	var results = new Array();
	
	while(true){
		results = inter_conn(sub,li,chk,1);
		sub = results[0].slice();
		li = results[1].slice();
		chk = results[2].slice();
		while(results.length > 0)	results.pop();
		results = inter_conn(sub,li,chk,-1);
		sub = results[0].slice();
		li = results[1].slice();
		chk = results[2].slice();
		while(results.length > 0)	results.pop();
		results = inter_conn(sub,li,chk,2);
		sub = results[0].slice();
		li = results[1].slice();
		chk = results[2].slice();
		while(results.length > 0)	results.pop();
		results = inter_conn(sub,li,chk,-2);
		sub = results[0].slice();
		li = results[1].slice();
		chk = results[2].slice();
		while(results.length > 0)	results.pop();
		
		inter.push(sub.slice());
		while(sub.length > 0)	sub.pop();
		
		var j = 0;
		while(j < li.length){
			if(chk[j] == 0){
				sub.push(li[j][0]);
				sub.push(li[j][1]);
				chk[j] = 1;
				break;
			}
			j += 1;
		}
		
		if(j == li.length)
			break;
	}
	return inter.slice();
}

function new_connection(arr){
	if(arr.length == 1)
		return arr[0];
	
	var a = -1;
	var b = -1;
	var c = -1;
	var d = 0;
	var tmpLi = new Array();
	var n = arr.length;
	for(var i = 0; i < n-1; i++){
		var min = Infinity;
		for(var j = 0; j < arr[0].length-1; j++)
			for(var k = 1; k < arr.length; k++)
				for(var l = 0; l < arr[k].length-1; l++){
					var dele = calDistance(arr[0][j],arr[0][j+1]) + calDistance(arr[k][l],arr[k][l+1]);
					
					var temp = calDistance(arr[0][j],arr[k][l]) + calDistance(arr[0][j+1],arr[k][l+1]);
					temp -= dele;
					
					if(min > temp){
						min = temp;
						a = j;
						b = k;
						c = l;
						d = 1;
					}
					
					temp = calDistance(arr[0][j],arr[k][l+1]) + calDistance(arr[0][j+1],arr[k][l]);
					temp -= dele;
					
					if(min > temp){
						min = temp;
						a = j;
						b = k;
						c = l;
						d = -1;
					}
				}				
		
		for(var j = 0; j < arr[0].length; j++){
			tmpLi.push(arr[0][j]);
			if(j == a){
				if(d == 1)
					for(var k = 0; k < arr[b].length-1; k++)
						tmpLi.push(arr[b][(c+arr[b].length-1-k)%(arr[b].length-1)]);
				else if(d == -1)
					for(var k = 0; k < arr[b].length-1; k++)
						tmpLi.push(arr[b][(c+k)%(arr[b].length-1)]);
			}
		}
		var idx = arr.indexOf(arr[b]);
		if(idx > -1) arr.splice(idx,1);
		idx = arr.indexOf(arr[0]);
		if(idx > -1) arr.splice(idx,1);
		
		arr.splice(0, 0, tmpLi.slice());
		while(tmpLi.length > 0)	tmpLi.pop();
	}
	return arr[0];
}

function show_cycle(){
	if(show_e){
		drawSub(esets);
		show_e = false;
	}
}

function show_inter(){
	if(show_i){
		drawInter(inter_edges,esets);
		show_i = false;
	}
}

function show_newGene(){
	if(show_n){
		drawPath(next_generation);
		show_n = false;
	}
	
	if(next_cost.length > 0)
		document.getElementById("costO1").innerHTML = "Cost (Offspring1): " + next_cost[0];
	if(next_cost.length > 1)
		document.getElementById("costO2").innerHTML = "Cost (Offspring2): " + next_cost[1];
	if(next_cost.length > 2)
		document.getElementById("costO3").innerHTML = "Cost (Offspring3): " + next_cost[2];
}

function title(){
	ctx.font = "14px Arial";
	ctx.fillStyle = "black";
	ctx.fillText('Parent A', 110,230);
	ctx.fillText('Parent B', 110,460);
	ctx.fillText('Generate AB', 110,710);
	
	ctx.fillText('E-set', 360,710);
	ctx.fillText('intermediates', 600,710);
	ctx.fillText('offsprings', 850,710);
	
	ctx.fillStyle = "red";
	ctx.fillText('(1)', 760,20);
	ctx.fillText('(2)', 760,230);
	ctx.fillText('(3)', 760,470);
}

function createCities(n){
	var r = 3;
	for(var i = 0; i < n; i++){
		var city = new Array();
		var randX = Math.floor(Math.random() * (180)) + 50;
		var randY = Math.floor(Math.random() * (180)) + 20;
		cities.push([randX,randY]);
		order.push(i);
	}
	order.push(0);
}

function keepCities(){
	var r = 3;
	title();
	for(var i = 0; i < totalCities; i++){
		var X = cities[i][0];
		var Y = cities[i][1];
		
		ctx.beginPath();
		ctx.arc(X, Y, r, 0, 2 * Math.PI);
		ctx.strokeStyle = 'black';
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X, Y + 230, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X, Y + 470, r, 0, 2 * Math.PI);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.arc(X + 250, Y, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 250, Y + 230, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 250, Y + 470, r, 0, 2 * Math.PI);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.arc(X + 490, Y, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 490, Y + 230, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 490, Y + 470, r, 0, 2 * Math.PI);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.arc(X + 740, Y, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 740, Y + 230, r, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.beginPath();
		ctx.arc(X + 740, Y + 470, r, 0, 2 * Math.PI);
		ctx.stroke();
	}
}

function mergePath(A, B){
	for(var i = 0; i < A.length-1; i++){
		ctx.beginPath();
		ctx.moveTo(cities[A[i]][0],cities[A[i]][1] + 470);
		ctx.lineTo(cities[A[i+1]][0],cities[A[i+1]][1] + 470);
		ctx.strokeStyle = 'red';
		ctx.stroke();
		
		ctx.beginPath();
		ctx.moveTo(cities[B[i]][0],cities[B[i]][1] + 470);
		ctx.lineTo(cities[B[i+1]][0],cities[B[i+1]][1] + 470);
		ctx.strokeStyle = 'green';
		ctx.stroke();
	}
}

function drawInter(arr, e){
	var intermd = new Array();
	for(var i = 0; i < e.length; i++){
		var temp = arr.slice();
		for(var j = 0; j < e[i].length; j++){
			for(var k = 0; k < e[i][j].length-1; k += 2){
				for(var l = 0; l < temp.length; l++){
					if(e[i][j][k] == temp[l][0])
						if(e[i][j][k+1] == temp[l][1]){
							var idx = temp.indexOf(temp[l]);
							if(idx > -1) temp.splice(idx,1);
							break;
						}
					if(e[i][j][k+1] == temp[l][0])
						if(e[i][j][k] == temp[l][1]){
							var idx = temp.indexOf(temp[l]);
							if(idx > -1) temp.splice(idx,1);
							break;
						}
				}
			}
			for(var k = 1; k < e[i][j].length; k += 2){
				edgeB = [e[i][j][k],e[i][j][k+1]];
				temp.push(edgeB.slice());
				while(edgeB.length > 0)	edgeB.pop();
			}
		}
		intermd.push(temp.slice());
		while(temp.length > 0)	temp.pop();
	}
	for(var i = 0; i < intermd.length; i++){
		if(i == 0)		z = 0;
		else if(i == 1)	z = 230;
		else if(i == 2)	z = 470;
		else			break;
		
		for(var j = 0; j < intermd[i].length; j++){
			ctx.beginPath();
			ctx.moveTo(cities[intermd[i][j][0]][0] + 490,cities[intermd[i][j][0]][1] + z);
			ctx.lineTo(cities[intermd[i][j][1]][0] + 490,cities[intermd[i][j][1]][1] + z);
			ctx.strokeStyle = 'black';
			ctx.stroke();
		}
	}
}

function drawPath(arr){
	for(var i = 0; i < arr.length; i++){
		if(i == 0)		z = 0;
		else if(i == 1)	z = 230;
		else if(i == 2)	z = 470;
		else			break;
		
		for(var j = 0; j < arr[i].length-1; j++){
			ctx.beginPath();
			ctx.moveTo(cities[arr[i][j]][0] + 740,cities[arr[i][j]][1] + z);
			ctx.lineTo(cities[arr[i][j+1]][0] + 740,cities[arr[i][j+1]][1] + z);
			ctx.strokeStyle = 'black';
			ctx.stroke();
		}
	}
}

function drawSub(arr){
	for(var i = 0; i < arr.length; i++){
		var toggle = 1;
		if(i == 0)		z = 0;
		else if(i == 1)	z = 230;
		else if(i == 2)	z = 470;
		else			break;
		
		for(var j = 0; j < arr[i].length; j++)
			for(var k = 0; k < arr[i][j].length-1; k++){
				if(toggle == 1){
					ctx.beginPath();
					ctx.moveTo(cities[arr[i][j][k]][0] + 250,cities[arr[i][j][k]][1] + z);
					ctx.lineTo(cities[arr[i][j][k+1]][0] + 250,cities[arr[i][j][k+1]][1] + z);
					ctx.strokeStyle = 'red';
					ctx.stroke();
				} else {
					ctx.beginPath();
					ctx.moveTo(cities[arr[i][j][k]][0] + 250,cities[arr[i][j][k]][1] + z);
					ctx.lineTo(cities[arr[i][j][k+1]][0] + 250,cities[arr[i][j][k+1]][1] + z);
					ctx.strokeStyle = 'green';
					ctx.stroke();
				}
				toggle *= -1;
			}
	}
}

function drawTwoPath(arr1,arr2){
	for(var i = 0; i < arr1.length-1; i++){
		ctx.beginPath();
		ctx.moveTo(cities[arr1[i]][0],cities[arr1[i]][1]);
		ctx.lineTo(cities[arr1[i+1]][0],cities[arr1[i+1]][1]);
		ctx.strokeStyle = 'red';
		ctx.stroke();
	
		ctx.beginPath();
		ctx.moveTo(cities[arr2[i]][0],cities[arr2[i]][1] + 230);
		ctx.lineTo(cities[arr2[i+1]][0],cities[arr2[i+1]][1] + 230);
		ctx.strokeStyle = 'green';
		ctx.stroke();
	}
}