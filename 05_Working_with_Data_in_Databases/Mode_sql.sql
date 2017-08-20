/*
Using t-test for two groups with different variances.
Do A/B test based on below user engagement event names
  -login
  -home_page   
  -like_message  
  -search_autocomplete
  -search_click_result_1
  -search_click_result_2
  -search_click_result_3
  -search_click_result_4
  -search_click_result_5
  -search_click_result_6
  -search_click_result_7
  -search_click_result_8
  -search_click_result_9
  -search_click_result_10
  -search_run
  -send_message
  -view_inbox
*/


select experiment, 
       experiment_group,
       event_name,
       event_type,
       sum(users)                                       as users,
       sum(total)                                       as total,
       cast(avg(total/users) as decimal(10,2))          as average,
       cast(sqrt(stddev_samp(total)) as decimal(10,6))  as stdev
from (
      select  exp.experiment, 
              exp.experiment_group,
              eve.event_name,
              eve.event_type,
    count(distinct exp.user_id)     as users,
    count(1)                        as total
      from tutorial.yammer_experiments exp
          join tutorial.yammer_events eve
          on   exp.user_id = eve.user_id
      where eve.occurred_at between to_date('2014-06-01 00:00:00','yyyy-mm-dd hh24:mi:ss') 
                                and to_date('2014-06-30 23:59:59','yyyy-mm-dd hh24:mi:ss')
      group by exp.experiment, 
          exp.experiment_group,
          exp.user_id,
          eve.event_name,
          eve.event_type
) sub
group by  experiment, 
          experiment_group,
          event_name,
          event_type
having    stddev_samp(total) > 0
order by  experiment, 
          event_name,
          event_type,
          experiment_group;



