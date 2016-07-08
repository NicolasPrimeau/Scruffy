$(function () {

    $.get("/analytics/get_stats", function( data ) {
        stats = JSON.parse(data);
        $("p#game_cnt_val").html(stats["count"]);
        $("p#max_score_val").html(stats["max"]);
    });

    $.get( "analytics/get_scores", function( data ) {

        var scores_preprocess = JSON.parse(data);
        var scores = [];
        var max = 0;
        for (var i=0; i<scores_preprocess.length; i+=1) {
            if (scores_preprocess[i].score > max) {
                max = scores_preprocess[i].score
            }
             $("p#max_val").html(max);
            scores.push([i, scores_preprocess[i].score]);
        }
        $.get( "analytics/get_fitted_line", function( data ) {
            var line = JSON.parse(data)
            line[0][0] = scores[0][0]
            line[1][0] = scores[scores.length-1][0]
            $('#container').highcharts({
                chart: {
                    type: 'scatter',
                    zoomType: 'xy'
                },
                title: {
                    text: 'Scores'
                },
                xAxis: {
                    title: {
                        enabled: true,
                        text: 'Time'
                    },
                    startOnTick: true,
                    endOnTick: true,
                    showLastLabel: true
                },
                yAxis: {
                    title: {
                        text: 'Score'
                    }
                },

                plotOptions: {
                    scatter: {
                        marker: {
                            radius: 5,
                            states: {
                                hover: {
                                    enabled: true,
                                    lineColor: 'rgb(100,100,100)'
                                }
                            }
                        },
                        states: {
                            hover: {
                                marker: {
                                    enabled: false
                                }
                            }
                        }
                    }
                },
                series: [{
                    name: 'Data',
                    color: 'rgba(223, 83, 83, .5)',
                    data: scores

                }, {
                type: 'line',
                name: 'Regression Line',
                data: line,
                marker: {
                    enabled: false
                },
                states: {
                    hover: {
                        lineWidth: 0
                    }
                },
                enableMouseTracking: false
                }]
            });
        });
    });
});
